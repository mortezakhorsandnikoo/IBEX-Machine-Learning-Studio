using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using ArchiAI.ML;

// LiveCharts2
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using LiveChartsCore.Defaults;
using SkiaSharp;

// NOTE: change this namespace to match your existing MLView (Views/MLView.xaml.cs).
namespace ArchiAI.Views
{
    // Third partial: the Visualization tab.
    // Chart types: Line, Bar, Scatter (2D), Pie, Radar/Spider.
    // Everything is driven by the LIVE _dataset, i.e. the current preprocessed data.
    public partial class MLView
    {
        private bool _vizSortAscending;

        private double[][] VisualizationRows() => (_vizSortAscending
            ? _dataset.Rows.OrderBy(row => row[XCol])
            : _dataset.Rows.AsEnumerable()).ToArray();

        private void ResetVisualization()
        {
            _vizSortAscending = false;
            VizSortButton.Content = "Sort X: min → max";
            VizCartesian.Series = Array.Empty<ISeries>();
            VizPie.Series = Array.Empty<ISeries>();
            VizPolar.Series = Array.Empty<ISeries>();
            VizStatus.Text = "Choose columns and plot your data. Source row order is used by default.";
        }

        private void VizSort_Click(object sender, RoutedEventArgs e)
        {
            if (_dataset == null) { VizStatus.Text = "Import a CSV from the Dataset page first."; return; }
            _vizSortAscending = !_vizSortAscending;
            VizSortButton.Content = _vizSortAscending ? "Restore source order" : "Sort X: min → max";
            VizPlot_Click(sender, e);
        }

        private void VizChartType_Changed(object sender, SelectionChangedEventArgs e)
        {
            if (VizSortButton == null) return;
            // Radar axes are feature names, not sortable numeric X observations.
            VizSortButton.IsEnabled = (VizChartTypeCombo.SelectedItem as ComboBoxItem)?.Content?.ToString() != "Radar";
        }

        // A small categorical palette for pie / multi-series.
        private static readonly SKColor[] VizPalette =
        {
            SKColor.Parse("#087F73"), SKColor.Parse("#5275C5"), SKColor.Parse("#ff6b6b"),
            SKColor.Parse("#5aa9e6"), SKColor.Parse("#b892ff"), SKColor.Parse("#7ae582"),
            SKColor.Parse("#ff9f68"), SKColor.Parse("#f18fb0"), SKColor.Parse("#4dd0e1"),
            SKColor.Parse("#c0ca33")
        };

        // Call from LoadCsv_Click / RefreshDataPreview so column pickers stay in sync
        // with the CURRENT (post-preprocessing) dataset.
        public void RefreshVizCombos()
        {
            if (VizXCombo == null) return;

            var headers = _dataset?.Headers ?? new string[0];

            FillColumnCombo(VizXCombo, headers, 0);
            FillColumnCombo(VizYCombo, headers, headers.Length > 1 ? 1 : 0);
            FillColumnCombo(VizY2Combo, headers, -1, addNone: true);
        }

        private static void FillColumnCombo(ComboBox combo, string[] headers, int defaultIndex, bool addNone = false)
        {
            if (combo == null) return;
            string previous = combo.SelectedItem?.ToString();
            combo.Items.Clear();
            if (addNone) combo.Items.Add("(none)");
            foreach (var h in headers) combo.Items.Add(h);
            int target = addNone ? defaultIndex + 1 : defaultIndex;
            if (previous != null && combo.Items.Contains(previous)) combo.SelectedItem = previous;
            else if (combo.Items.Count > 0)
                combo.SelectedIndex = target >= 0 && target < combo.Items.Count ? target : 0;
        }

        // Fires whenever the Visualization panel becomes visible. Repopulates the
        // X / Y / Y2 pickers from the CURRENT preprocessed dataset.
        private void VizPanel_IsVisibleChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            if (e.NewValue is bool visible && visible) RefreshVizCombos();
        }

        private void VizPlot_Click(object sender, RoutedEventArgs e)
        {
            if (_dataset == null || _dataset.RowCount == 0)
            {
                VizStatus.Text = "Load a CSV first.";
                return;
            }

            string type = (VizChartTypeCombo.SelectedItem as ComboBoxItem)?.Content?.ToString()
                          ?? VizChartTypeCombo.SelectedItem?.ToString() ?? "Line";

            // Toggle which chart control is visible.
            VizCartesian.Visibility = (type == "Pie" || type == "Radar") ? Visibility.Collapsed : Visibility.Visible;
            VizPie.Visibility = (type == "Pie") ? Visibility.Visible : Visibility.Collapsed;
            VizPolar.Visibility = (type == "Radar") ? Visibility.Visible : Visibility.Collapsed;

            try
            {
                switch (type)
                {
                    case "Line": PlotLine(); break;
                    case "Bar": PlotBar(); break;
                    case "Scatter": PlotScatter(); break;
                    case "Pie": PlotPie(); break;
                    case "Radar": PlotRadar(); break;
                }
                string order = type == "Radar" ? "Feature order; original row index"
                    : _vizSortAscending ? "X sorted from min to max" : type == "Pie" ? "Largest slices first" : "Source row order";
                VizStatus.Text = $"{type} chart · {_dataset.RowCount} rows · {order}";
                if (type == "Bar" && _dataset.RowCount > 40) VizStatus.Text += " · First 40 plotted rows";
                if (type == "Pie" && _dataset.RowCount > 10) VizStatus.Text += " · Top 10 values + Other";
            }
            catch (Exception ex)
            {
                VizStatus.Text = "Plot failed: " + ex.Message;
            }
        }

        private int XCol => Math.Max(0, VizXCombo.SelectedIndex);
        private int YCol => Math.Max(0, VizYCombo.SelectedIndex);
        // VizY2Combo has a leading "(none)" item, so index 0 == no second series.
        private int Y2Col => VizY2Combo.SelectedIndex - 1;

        private void PlotLine()
        {
            var rows = VisualizationRows();
            var pts = rows.Select(row => new ObservablePoint(row[XCol], row[YCol])).ToArray();

            var series = new List<ISeries>
            {
                new LineSeries<ObservablePoint>
                {
                    Values = pts, Name = _dataset.Headers[YCol],
                    Stroke = new SolidColorPaint(VizPalette[0], 2), Fill = null,
                    GeometryFill = new SolidColorPaint(VizPalette[0]), GeometryStroke = null,
                    GeometrySize = 6, LineSmoothness = 0.2
                }
            };

            if (Y2Col >= 0)
            {
                var ordered2 = rows.Select(row => new ObservablePoint(row[XCol], row[Y2Col])).ToArray();
                series.Add(new LineSeries<ObservablePoint>
                {
                    Values = ordered2,
                    Name = _dataset.Headers[Y2Col],
                    Stroke = new SolidColorPaint(VizPalette[2], 2),
                    Fill = null,
                    GeometryFill = new SolidColorPaint(VizPalette[2]),
                    GeometryStroke = null,
                    GeometrySize = 6,
                    LineSmoothness = 0.2
                });
            }

            VizCartesian.Series = series.ToArray();
            VizCartesian.XAxes = MakeAxes(_dataset.Headers[XCol]);
            VizCartesian.YAxes = MakeAxes(_dataset.Headers[YCol]);
        }

        private void PlotScatter()
        {
            var pts = VisualizationRows().Select(row => new ObservablePoint(row[XCol], row[YCol])).ToArray();

            VizCartesian.Series = new ISeries[]
            {
                new ScatterSeries<ObservablePoint>
                {
                    Values = pts, Name = $"{_dataset.Headers[XCol]} vs {_dataset.Headers[YCol]}",
                    Fill = new SolidColorPaint(VizPalette[0]), GeometrySize = 8
                }
            };
            VizCartesian.XAxes = MakeAxes(_dataset.Headers[XCol]);
            VizCartesian.YAxes = MakeAxes(_dataset.Headers[YCol]);
        }

        private void PlotBar()
        {
            var rows = VisualizationRows().Take(40).ToArray();
            var vals = rows.Select(row => row[YCol]).ToArray();
            var labels = rows.Select(row => row[XCol].ToString("0.##")).ToList();

            VizCartesian.Series = new ISeries[]
            {
                new ColumnSeries<double>
                {
                    Values = vals, Name = _dataset.Headers[YCol],
                    Fill = new SolidColorPaint(VizPalette[0])
                }
            };
            VizCartesian.XAxes = new Axis[]
            {
                new Axis
                {
                    Name = _dataset.Headers[XCol],
                    Labels = labels, LabelsRotation = 45,
                    NamePaint = new SolidColorPaint(PolyText),
                    LabelsPaint = new SolidColorPaint(PolyText),
                    SeparatorsPaint = new SolidColorPaint(PolyGrid)
                }
            };
            VizCartesian.YAxes = MakeAxes(_dataset.Headers[YCol]);
        }

        private void PlotPie()
        {
            var y = _dataset.GetColumn(YCol).Select(Math.Abs).ToArray();
            var x = _dataset.GetColumn(XCol);

            // Too many slices are unreadable: keep top 10 by value, bucket the rest as "Other".
            var indexed = y.Select((v, i) => new { v, category = x[i], label = x[i].ToString("0.##") })
                           .OrderByDescending(p => p.v).ToList();

            int top = Math.Min(10, indexed.Count);
            var slices = indexed.Take(top).ToList();
            if (_vizSortAscending) slices = slices.OrderBy(p => p.category).ToList();
            double other = indexed.Skip(top).Sum(p => p.v);

            var series = new List<ISeries>();
            for (int i = 0; i < slices.Count; i++)
            {
                string label = slices[i].label; // capture per-iteration for the closure
                series.Add(new PieSeries<double>
                {
                    Values = new double[] { slices[i].v },
                    Name = label,
                    Fill = new SolidColorPaint(VizPalette[i % VizPalette.Length]),
                    DataLabelsPaint = new SolidColorPaint(PolyText),
                    DataLabelsPosition = LiveChartsCore.Measure.PolarLabelsPosition.Outer,
                    DataLabelsFormatter = point => label
                });
            }
            if (other > 0)
                series.Add(new PieSeries<double>
                {
                    Values = new double[] { other },
                    Name = "Other",
                    Fill = new SolidColorPaint(PolyGrid),
                    DataLabelsPaint = new SolidColorPaint(PolyText)
                });

            VizPie.Series = series.ToArray();
        }

        // Radar / spider: one axis per numeric column, min-max normalized to 0..1.
        // Series 1 = column means (the "average profile").
        // Series 2 = a selected row (VizRowBox), to compare a data point against the mean.
        private void PlotRadar()
        {
            int cols = _dataset.ColCount;
            var headers = _dataset.Headers.ToList();

            double[] min = new double[cols], max = new double[cols], mean = new double[cols];
            for (int c = 0; c < cols; c++)
            {
                var col = _dataset.GetColumn(c);
                min[c] = col.Min();
                max[c] = col.Max();
                mean[c] = col.Average();
            }

            double Norm(int c, double v)
            {
                double range = max[c] - min[c];
                return range < 1e-12 ? 0.5 : (v - min[c]) / range;
            }

            var meanNorm = Enumerable.Range(0, cols).Select(c => Norm(c, mean[c])).ToArray();

            var series = new List<ISeries>
            {
                new PolarLineSeries<double>
                {
                    Values = meanNorm, Name = "Mean",
                    Stroke = new SolidColorPaint(PolyAccent, 2),
                    Fill = new SolidColorPaint(PolyAccent.WithAlpha(60)),
                    GeometrySize = 6, IsClosed = true
                }
            };

            int rowIdx = ParseInt(VizRowBox?.Text, 0, 0, _dataset.RowCount - 1);
            if (_dataset.RowCount > 0)
            {
                var row = _dataset.Rows[rowIdx];
                var rowNorm = Enumerable.Range(0, cols).Select(c => Norm(c, row[c])).ToArray();
                series.Add(new PolarLineSeries<double>
                {
                    Values = rowNorm,
                    Name = $"Row {rowIdx}",
                    Stroke = new SolidColorPaint(PolyYellow, 2),
                    Fill = new SolidColorPaint(PolyYellow.WithAlpha(50)),
                    GeometrySize = 6,
                    IsClosed = true
                });
            }

            VizPolar.Series = series.ToArray();
            VizPolar.AngleAxes = new PolarAxis[]
            {
                new PolarAxis
                {
                    Labels = headers,
                    LabelsPaint = new SolidColorPaint(PolyText),
                    SeparatorsPaint = new SolidColorPaint(PolyGrid)
                }
            };
            VizPolar.RadiusAxes = new PolarAxis[]
            {
                new PolarAxis
                {
                    MinLimit = 0, MaxLimit = 1,
                    LabelsPaint = new SolidColorPaint(PolyText),
                    SeparatorsPaint = new SolidColorPaint(PolyGrid)
                }
            };
        }
    }
}