using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using SkiaSharp;

namespace ArchiAI.Views
{
    // Paper-ready export for the Machine Learning tabs (matches the Optimization view):
    // Save-diagrams / Save-tables act on the CURRENT tab, with adjustable label/title/number sizes
    // and line width. Charts render headless (transparent background, black text).
    public partial class MLView
    {
        // charts on each tab, by MainTabs.SelectedIndex
        private (string name, FrameworkElement el)[] ChartsForTab(int t)
        {
            switch (t)
            {
                case 0: return new (string, FrameworkElement)[] { ("histogram", HistChart), ("correlation_matrix", CorrCanvas) };
                case 1: return new (string, FrameworkElement)[] { ("missing", MissingChart), ("anomaly", AnomalyChart), ("normalize", NormChart), ("scree", ScreeChart), ("cumulative_variance", CumVarChart), ("pca_scatter", PcaScatterChart) };
                case 2: return new (string, FrameworkElement)[] { ("cost", CostChart), ("train_scatter", TrainScatter), ("test_scatter", TestScatter), ("residual", ResidualChart), ("importance", ImportanceChart), ("poly_cost", PolyCostChart), ("poly_scatter", PolyScatterChart), ("poly_residual", PolyResidualChart), ("poly_importance", PolyImportanceChart), ("adv_scatter", AdvScatter), ("adv_residual", AdvResidual), ("adv_learning", AdvLearnChart), ("adv_validation", AdvValChart), ("tree", AdvTreeCanvas) };
                case 3: return new (string, FrameworkElement)[] { ("f1", ClsF1Chart), ("confusion_matrix", ClsConfCanvas) };
                case 4: return new (string, FrameworkElement)[] { ("cluster_scatter", ClustScatter) };
                case 5: return new (string, FrameworkElement)[] { ("compare_r2", CompareRegChart), ("compare_rmse", CompareRmseChart), ("compare_classification", CompareClsChart) };
                case 6: return new (string, FrameworkElement)[] { ("visualization", VizCartesian), ("pie", VizPie), ("radar", VizPolar) };
                default: return new (string, FrameworkElement)[0];
            }
        }

        private (string name, DataGrid grid)[] TablesForTab(int t)
        {
            switch (t)
            {
                case 0: return new (string, DataGrid)[] { ("data_preview", DataPreview) };
                case 1: return new (string, DataGrid)[] { ("describe", DescribeGrid), ("normalize", NormalizeGrid), ("encoded", EncoderGrid) };
                case 3: return new (string, DataGrid)[] { ("classification_metrics", ClsMetricsGrid) };
                case 5: return new (string, DataGrid)[] { ("model_comparison", CompareGrid) };
                default: return new (string, DataGrid)[0];
            }
        }

        private void MlSavePage_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new Microsoft.Win32.SaveFileDialog { Filter = "PNG image (*.png)|*.png", FileName = "ibex_page.png" };
            if (dlg.ShowDialog() != true) return;
            try { MlSaveWpfTransparent(MainTabs, dlg.FileName, 2); ExportStatus.Text = "Page image saved (current view)."; }
            catch (Exception ex) { ExportStatus.Text = "Could not save page: " + ex.Message; }
        }

        private void MlSaveDiagrams_Click(object sender, RoutedEventArgs e)
        {
            var charts = ExportableCharts();
            if (charts.Length == 0) { ExportStatus.Text = "No charts yet. Plot or run an analysis first."; return; }
            var dlg = new Microsoft.Win32.SaveFileDialog { Filter = "PNG image (*.png)|*.png", FileName = "chart.png" };
            if (dlg.ShowDialog() != true) return;
            string dir = System.IO.Path.GetDirectoryName(dlg.FileName), b = System.IO.Path.GetFileNameWithoutExtension(dlg.FileName);
            int saved = 0;
            var errors = new List<string>();
            foreach (var (name, el) in charts)
                try { MlExportChart(el, System.IO.Path.Combine(dir, $"{b}_{name}.png")); saved++; }
                catch (Exception ex) { errors.Add(name + ": " + ex.Message); }
            ExportStatus.Text = $"Saved {saved} chart/image(s)." + (errors.Count > 0 ? " Failed: " + string.Join("; ", errors) : "");
        }

        private void MlSaveTables_Click(object sender, RoutedEventArgs e)
        {
            var tables = BuildExportTables();
            if (tables.Count == 0) { ExportStatus.Text = "No results yet. Import data or run an analysis first."; return; }
            var dlg = new Microsoft.Win32.SaveFileDialog { Filter = "CSV (*.csv)|*.csv", FileName = "table.csv" };
            if (dlg.ShowDialog() != true) return;
            string dir = System.IO.Path.GetDirectoryName(dlg.FileName), b = System.IO.Path.GetFileNameWithoutExtension(dlg.FileName);
            int saved = 0;
            var errors = new List<string>();
            foreach (var table in tables)
                try { System.IO.File.WriteAllText(System.IO.Path.Combine(dir, $"{b}_{table.Key}.csv"), table.Value, new UTF8Encoding(true)); saved++; }
                catch (Exception ex) { errors.Add(table.Key + ": " + ex.Message); }
            ExportStatus.Text = $"Saved {saved} table(s)." + (errors.Count > 0 ? " Failed: " + string.Join("; ", errors) : "");
        }

        // ── export helpers (same approach as the Optimization view) ──
        private void MlExportChart(FrameworkElement el, string path)
        {
            const double scale = 4.0;
            int w = Math.Max(2, (int)(el.ActualWidth * scale));
            int h = Math.Max(2, (int)(el.ActualHeight * scale));
            double font = 30, titleFont = 36, numFont = 28, lineW = 6;
            if (MlLabelBox != null && double.TryParse(MlLabelBox.Text, out double f) && f > 4) font = f;
            if (MlTitleBox != null && double.TryParse(MlTitleBox.Text, out double tf) && tf > 4) titleFont = tf;
            if (MlNumBox != null && double.TryParse(MlNumBox.Text, out double nf) && nf > 2) numFont = nf;
            if (MlLineBox != null && double.TryParse(MlLineBox.Text, out double lw) && lw > 0) lineW = lw;

            if (el is LiveChartsCore.SkiaSharpView.WPF.CartesianChart cc) { MlExportCartesian(cc, path, w, h, font, titleFont, numFont, lineW); return; }
            if (el is LiveChartsCore.SkiaSharpView.WPF.PieChart pie)
            {
                var sizes = pie.Series.OfType<LiveChartsCore.SkiaSharpView.PieSeries<double>>().Select(series => (series, series.DataLabelsSize)).ToArray();
                try
                {
                    foreach (var (series, _) in sizes) series.DataLabelsSize = numFont;
                    var sk = new LiveChartsCore.SkiaSharpView.SKCharts.SKPieChart { Width = w, Height = h, Background = SKColors.White, Series = pie.Series };
                    sk.SaveImage(path);
                }
                finally { foreach (var (series, size) in sizes) series.DataLabelsSize = size; }
                return;
            }
            if (el is LiveChartsCore.SkiaSharpView.WPF.PolarChart pc) { MlExportPolar(pc, path, w, h); return; }
            MlSaveWpfTransparent(el, path, scale);
        }

        private static Axis MlBlackAxis(Axis a, double font, double titleFont) => new Axis
        {
            Name = a.Name, Labels = a.Labels, LabelsRotation = a.LabelsRotation,
            MinLimit = a.MinLimit, MaxLimit = a.MaxLimit, TextSize = font, NameTextSize = titleFont,
            LabelsPaint = new SolidColorPaint(SKColors.Black), NamePaint = new SolidColorPaint(SKColors.Black),
            SeparatorsPaint = new SolidColorPaint(SKColor.Parse("#DDDDDD"))
        };

        private static void MlExportCartesian(LiveChartsCore.SkiaSharpView.WPF.CartesianChart c, string path, int w, int h, double font, double titleFont, double numFont, double lineW)
        {
            var xs = (c.XAxes != null) ? c.XAxes.OfType<Axis>().Select(a => MlBlackAxis(a, font, titleFont)).ToArray() : null;
            var ys = (c.YAxes != null) ? c.YAxes.OfType<Axis>().Select(a => MlBlackAxis(a, font, titleFont)).ToArray() : null;
            var savedDl = new List<(object s, System.Reflection.PropertyInfo pi, object old)>();
            var savedStroke = new List<(object p, System.Reflection.PropertyInfo pi, object old)>();
            if (c.Series != null)
                foreach (var s in c.Series)
                {
                    var pi = s.GetType().GetProperty("DataLabelsSize");
                    if (pi != null && pi.CanWrite) { try { savedDl.Add((s, pi, pi.GetValue(s))); pi.SetValue(s, Convert.ChangeType(numFont, pi.PropertyType)); } catch { } }
                    try
                    {
                        var sp = s.GetType().GetProperty("Stroke")?.GetValue(s);
                        var st = sp?.GetType().GetProperty("StrokeThickness");
                        if (sp != null && st != null && st.CanWrite) { savedStroke.Add((sp, st, st.GetValue(sp))); st.SetValue(sp, Convert.ChangeType(lineW, st.PropertyType)); }
                    }
                    catch { }
                }
            try
            {
                var sk = new LiveChartsCore.SkiaSharpView.SKCharts.SKCartesianChart { Width = w, Height = h, Background = SKColors.Transparent, Series = c.Series };
                if (xs != null && xs.Length > 0) sk.XAxes = xs;
                if (ys != null && ys.Length > 0) sk.YAxes = ys;
                sk.SaveImage(path);
            }
            finally
            {
                foreach (var (s, pi, old) in savedDl) { try { pi.SetValue(s, old); } catch { } }
                foreach (var (p, pi, old) in savedStroke) { try { pi.SetValue(p, old); } catch { } }
            }
        }

        private static void MlExportPolar(LiveChartsCore.SkiaSharpView.WPF.PolarChart c, string path, int w, int h)
        {
            var sk = new LiveChartsCore.SkiaSharpView.SKCharts.SKPolarChart { Width = w, Height = h, Background = SKColors.Transparent, Series = c.Series };
            if (c.AngleAxes != null) sk.AngleAxes = c.AngleAxes;
            if (c.RadiusAxes != null) sk.RadiusAxes = c.RadiusAxes;
            sk.SaveImage(path);
        }

        internal static void MlSaveWpfTransparent(FrameworkElement fe, string path, double scale)
        {
            fe.UpdateLayout();
            int w = Math.Max(1, (int)(fe.ActualWidth * scale));
            int h = Math.Max(1, (int)(fe.ActualHeight * scale));
            var rtb = new RenderTargetBitmap(w, h, 96 * scale, 96 * scale, PixelFormats.Pbgra32);
            // A visual brush draws in local coordinates, avoiding the element's scroll/layout offset.
            var visual = new DrawingVisual();
            using (var dc = visual.RenderOpen())
            {
                var bounds = new Rect(0, 0, fe.ActualWidth, fe.ActualHeight);
                if (fe is Canvas canvas)
                {
                    // Matrix/tree drawings can be outside a ScrollViewer viewport. Reuse their
                    // vector drawings directly so ancestor clipping cannot cut the exported image.
                    dc.DrawRectangle(canvas.Background ?? Brushes.White, null, bounds);
                    foreach (UIElement child in canvas.Children)
                        for (int i = 0; i < VisualTreeHelper.GetChildrenCount(child); i++)
                            if (VisualTreeHelper.GetChild(child, i) is DrawingVisual drawing)
                                dc.DrawDrawing(drawing.Drawing);
                }
                else
                {
                    var brush = new VisualBrush(fe) { ViewboxUnits = BrushMappingMode.Absolute, Viewbox = bounds, Stretch = Stretch.Fill };
                    dc.DrawRectangle(brush, null, bounds);
                }
            }
            rtb.Render(visual);
            var enc = new PngBitmapEncoder();
            enc.Frames.Add(BitmapFrame.Create(rtb));
            using (var fs = System.IO.File.Create(path)) enc.Save(fs);
        }

        private static string MlGridToCsv(DataGrid grid)
        {
            string Q(string s) => (s != null && (s.Contains(",") || s.Contains("\"") || s.Contains("\n"))) ? "\"" + s.Replace("\"", "\"\"") + "\"" : (s ?? "");
            var sb = new StringBuilder();
            if (grid.ItemsSource is System.Data.DataView dv)
            {
                var cols = dv.Table.Columns.Cast<System.Data.DataColumn>().ToList();
                sb.AppendLine(string.Join(",", cols.Select(c => Q(c.ColumnName))));
                foreach (System.Data.DataRowView rv in dv) sb.AppendLine(string.Join(",", cols.Select(c => Q(rv[c.ColumnName]?.ToString()))));
            }
            else
            {
                var items = grid.ItemsSource.Cast<object>().ToList();
                if (items.Count > 0)
                {
                    var props = items[0].GetType().GetProperties();
                    sb.AppendLine(string.Join(",", props.Select(p => Q(p.Name))));
                    foreach (var it in items) sb.AppendLine(string.Join(",", props.Select(p => Q(p.GetValue(it)?.ToString()))));
                }
            }
            return sb.ToString();
        }
    }
}
