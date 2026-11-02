using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using LiveChartsCore;
using LiveChartsCore.Defaults;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.WPF;

namespace ArchiAI.Views
{
    public partial class MLView
    {
        private static IEnumerable<ISeries> ChartSeries(FrameworkElement element) => element switch
        {
            CartesianChart c => c.Series ?? Array.Empty<ISeries>(),
            PieChart c => c.Series ?? Array.Empty<ISeries>(),
            PolarChart c => c.Series ?? Array.Empty<ISeries>(),
            _ => Array.Empty<ISeries>()
        };

        private (string name, FrameworkElement el)[] ExportableCharts() => ChartsForTab(MainTabs.SelectedIndex)
            .Where(c => c.el.IsVisible && c.el.ActualWidth > 2 && c.el.ActualHeight > 2 &&
                (c.el is Canvas canvas ? canvas.Children.Count > 0 : ChartSeries(c.el).Any(s => s.Values?.Cast<object>().Any() == true)))
            .ToArray();

        private static string CsvValue(object value)
        {
            string text = Convert.ToString(value, CultureInfo.InvariantCulture) ?? "";
            return "\"" + text.Replace("\"", "\"\"") + "\"";
        }

        private static string CsvRows(IEnumerable<IEnumerable<object>> rows) => string.Join("\r\n", rows.Select(row => string.Join(",", row.Select(CsvValue)))) + "\r\n";

        // Includes numeric plot data on pages without a DataGrid, in the exact plotted order.
        public Dictionary<string, string> BuildExportTables()
        {
            var output = new Dictionary<string, string>();
            var cv = ActiveCrossValidationTable()?.Report;
            if (cv != null)
            {
                string metric = cv.Classification ? "accuracy (%)" : "R2";
                output["cross_validation"] = CsvRows(new[] { new object[] { "Fold", "Train rows", "Test rows", "Train " + metric, "Test " + metric } }
                    .Concat(cv.Rows.Select(r => new object[] { r.Fold, r.TrainRows, r.TestRows,
                        double.IsFinite(r.TrainScore) ? (object)r.TrainScore : "N/A",
                        double.IsFinite(r.TestScore) ? (object)r.TestScore : "N/A" })));
            }
            foreach (var (name, grid) in TablesForTab(MainTabs.SelectedIndex))
                if (grid.ItemsSource != null) output[name] = MlGridToCsv(grid);
            foreach (var (name, element) in ExportableCharts())
            {
                var rows = new List<object[]> { new object[] { "Series", "Index", "Category", "X", "Y" } };
                string[] labels = element switch
                {
                    CartesianChart c => c.XAxes?.OfType<Axis>().FirstOrDefault()?.Labels?.ToArray(),
                    PolarChart c => c.AngleAxes?.OfType<PolarAxis>().FirstOrDefault()?.Labels?.ToArray(),
                    _ => null
                };
                foreach (var series in ChartSeries(element))
                {
                    int index = 0;
                    foreach (object value in series.Values ?? Array.Empty<object>())
                    {
                        string label = labels != null && index < labels.Length ? labels[index] : "";
                        if (value is ObservablePoint point)
                            rows.Add(new object[] { series.Name, index, label, point.X, point.Y });
                        else rows.Add(new object[] { series.Name, index, label, index, value });
                        index++;
                    }
                }
                if (rows.Count > 1) output[name + "_data"] = CsvRows(rows);
            }
            if (_dataset != null && DataPreview.ItemsSource != null && MainTabs.SelectedIndex == 0)
            {
                var rows = new List<object[]> { new object[] { "Column" }.Concat(_dataset.Headers.Cast<object>()).ToArray() };
                var matrix = _dataset.CorrelationMatrix();
                for (int i = 0; i < _dataset.ColCount; i++)
                    rows.Add(new object[] { _dataset.Headers[i] }.Concat(Enumerable.Range(0, _dataset.ColCount).Select(j => (object)matrix[i, j])).ToArray());
                output["correlation_matrix"] = CsvRows(rows);
                output["dataset"] = CsvRows(new[] { _dataset.Headers.Cast<object>() }.Concat(_dataset.Rows.Select(row => row.Cast<object>())));
            }
            if (MainTabs.SelectedIndex == 2 && ExportableCharts().Length > 0)
            {
                string metrics = CmbRegAlgo.SelectedIndex == 0 ? TxtResults.Text : CmbRegAlgo.SelectedIndex == 1 ? PolyMetricsText.Text : TxtAdvResults.Text;
                output["metrics"] = CsvRows(new[] { new object[] { "Result" } }.Concat(metrics.Split('\n').Where(line => !string.IsNullOrWhiteSpace(line)).Select(line => new object[] { line.Trim() })));
            }
            if (MainTabs.SelectedIndex == 3 && _clsResult?.Confusion != null)
            {
                var labels = _clsResult.Classes;
                var rows = new List<object[]> { new object[] { "Actual / Predicted" }.Concat(labels.Cast<object>()).ToArray() };
                for (int i = 0; i < labels.Length; i++)
                    rows.Add(new object[] { labels[i] }.Concat(Enumerable.Range(0, labels.Length).Select(j => (object)_clsResult.Confusion[i, j])).ToArray());
                output["confusion_matrix"] = CsvRows(rows);
            }
            if (MainTabs.SelectedIndex == 4 && _clustResult?.Labels != null)
                output["cluster_assignments"] = CsvRows(new[] { new object[] { "Row", "Cluster" } }.Concat(_clustResult.Labels.Select((label, index) => new object[] { index, label })));
            return output;
        }
    }
}
