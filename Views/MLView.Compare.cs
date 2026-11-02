using ArchiAI.ML;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using SkiaSharp;
using System;
using System.Data;
using System.Linq;
using System.Windows;

namespace ArchiAI.Views
{
    public partial class MLView
    {
        private void ComparePanel_IsVisibleChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            if ((bool)e.NewValue) RefreshCompare();
        }
        private void CompareRefresh_Click(object sender, RoutedEventArgs e) => RefreshCompare();
        private void CompareClear_Click(object sender, RoutedEventArgs e) { MLRunHistory.Clear(); RefreshCompare(); }

        private void RefreshCompare()
        {
            var runs = MLRunHistory.Runs;
            if (runs.Count == 0)
            {
                TxtCompareInfo.Text = ""; CompareGrid.ItemsSource = null;
                CompareRegChart.Series = CompareRmseChart.Series = CompareClsChart.Series = Array.Empty<ISeries>();
                return;
            }
            TxtCompareInfo.Text = $"{runs.Count} run(s) recorded this session";

            // table
            var dt = new DataTable();
            dt.Columns.Add("Task"); dt.Columns.Add("Algorithm"); dt.Columns.Add("Target"); dt.Columns.Add("Time");
            dt.Columns.Add("R²"); dt.Columns.Add("AdjR²"); dt.Columns.Add("MAE"); dt.Columns.Add("RMSE");
            dt.Columns.Add("Accuracy%"); dt.Columns.Add("MacroF1");
            string F(double v, string fmt) => double.IsNaN(v) ? "" : v.ToString(fmt);
            foreach (var r in runs)
            {
                var row = dt.NewRow();
                row[0] = r.Task; row[1] = r.Algorithm; row[2] = r.Target; row[3] = r.When;
                row[4] = F(r.R2, "F4"); row[5] = F(r.AdjR2, "F4"); row[6] = F(r.MAE, "G5"); row[7] = F(r.RMSE, "G5");
                row[8] = F(r.Accuracy, "F2"); row[9] = F(r.MacroF1, "F3");
                dt.Rows.Add(row);
            }
            CompareGrid.ItemsSource = dt.DefaultView;

            // regression charts
            var reg = runs.Where(r => r.Task == "Regression").ToList();
            var regLabels = reg.Select((r, i) => $"{i + 1}.{r.Algorithm}").ToList();
            CompareRegChart.Series = new ISeries[] { new ColumnSeries<double> { Values = reg.Select(r => double.IsNaN(r.R2) ? 0 : r.R2).ToList(), Fill = new SolidColorPaint(SKColor.Parse("#087F73")), Stroke = null, MaxBarWidth = 40 } };
            CompareRegChart.XAxes = new Axis[] { new Axis { Labels = regLabels, LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10, LabelsRotation = 25 } };
            CompareRegChart.YAxes = new Axis[] { new Axis { LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 } };

            CompareRmseChart.Series = new ISeries[] { new ColumnSeries<double> { Values = reg.Select(r => double.IsNaN(r.RMSE) ? 0 : r.RMSE).ToList(), Fill = new SolidColorPaint(SKColor.Parse("#ff6b6b")), Stroke = null, MaxBarWidth = 40 } };
            CompareRmseChart.XAxes = new Axis[] { new Axis { Labels = regLabels, LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10, LabelsRotation = 25 } };
            CompareRmseChart.YAxes = new Axis[] { new Axis { LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 } };

            // classification chart (Accuracy% + MacroF1*100)
            var cls = runs.Where(r => r.Task == "Classification").ToList();
            var clsLabels = cls.Select((r, i) => $"{i + 1}.{r.Algorithm}").ToList();
            CompareClsChart.Series = new ISeries[]
            {
                new ColumnSeries<double> { Name = "Accuracy%", Values = cls.Select(r => double.IsNaN(r.Accuracy) ? 0 : r.Accuracy).ToList(), Fill = new SolidColorPaint(SKColor.Parse("#087F73")), MaxBarWidth = 24 },
                new ColumnSeries<double> { Name = "MacroF1×100", Values = cls.Select(r => double.IsNaN(r.MacroF1) ? 0 : r.MacroF1 * 100).ToList(), Fill = new SolidColorPaint(SKColor.Parse("#5275C5")), MaxBarWidth = 24 }
            };
            CompareClsChart.XAxes = new Axis[] { new Axis { Labels = clsLabels, LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10, LabelsRotation = 25 } };
            CompareClsChart.YAxes = new Axis[] { new Axis { LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10, MinLimit = 0, MaxLimit = 105 } };
        }
    }
}
