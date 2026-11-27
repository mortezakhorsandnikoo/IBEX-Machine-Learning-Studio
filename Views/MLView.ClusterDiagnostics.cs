using ArchiAI.ML;
using LiveChartsCore;
using LiveChartsCore.Defaults;
using LiveChartsCore.Kernel;
using LiveChartsCore.Kernel.Sketches;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;

namespace ArchiAI.Views
{
    // Two-step clustering workflow:
    //   K-Means      → Show elbow method → pick k on the chart (or type it) → Run clustering.
    //   Hierarchical → Show dendrogram   → click a cut height (or type k)   → Run clustering.
    public partial class MLView
    {
        private (int k, double wcss)[] _elbowCurve;
        private MLDataset _elbowDataset;          // dataset the elbow curve was computed on
        private Dendrogram _dendrogram;
        private MLDataset _dendroDataset;         // dataset + linkage the tree was built for
        private bool _elbowClickHooked;

        private bool ElbowReady => _elbowCurve != null && ReferenceEquals(_elbowDataset, _dataset);
        private bool DendrogramReady => _dendrogram != null && ReferenceEquals(_dendroDataset, _dataset)
            && _dendrogram.Linkage == SelectedLinkage();

        private string SelectedLinkage() => (CmbClustLinkage?.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "average";

        private int CurrentClustK() => int.TryParse(TxtClustK?.Text, out int k) && k > 0 ? k : -1;

        // ---------------- Elbow method (K-Means) ----------------
        private async void ClustElbow_Click(object sender, RoutedEventArgs e)
        {
            if (_dataset == null) { TxtClustResults.Text = "Load data first!"; return; }
            if (!int.TryParse(TxtElbowMaxK.Text, out int kMax) || kMax < 2) { TxtClustResults.Text = "Enter a largest k of 2 or more for the elbow method."; return; }
            kMax = Math.Min(kMax, _dataset.RowCount);
            var data = _dataset;
            int resetVersion = _pageResetVersions[4];
            var engine = new ClusteringEngine { Type = ClustererType.KMeans };
            if (int.TryParse(TxtClustIter.Text, out int it) && it > 0) engine.Iterations = it;
            SetClustBusy(true, "Computing the elbow curve...");
            try
            {
                var curve = await Task.Run(() => engine.ElbowCurve(data, kMax));
                if (resetVersion != _pageResetVersions[4] || !ReferenceEquals(data, _dataset)) return;
                _elbowCurve = curve; _elbowDataset = data;
                int suggested = ClusteringEngine.SuggestElbow(curve);
                TxtClustK.Text = suggested.ToString(CultureInfo.InvariantCulture); // also redraws the selection
                TxtElbowInfo.Text = $"Suggested elbow: k = {suggested}. Click a point to choose another k, then run clustering.";
                ClustElbowBorder.Visibility = Visibility.Visible;
                TxtClustResults.Text = $"Elbow curve ready (k = 1…{curve.Length}). Step 2: choose k, then click RUN CLUSTERING.";
                DrawElbow();
            }
            catch (Exception ex) { if (resetVersion == _pageResetVersions[4]) TxtClustResults.Text = "❌ Elbow method: " + ex.Message; }
            finally { SetClustBusy(false); UpdateClustStepHint(); }
        }

        private void DrawElbow()
        {
            if (_elbowCurve == null) return;
            if (!_elbowClickHooked) { ClustElbowChart.DataPointerDown += ElbowChart_DataPointerDown; _elbowClickHooked = true; }
            int k = CurrentClustK();
            var series = new List<ISeries>
            {
                new LineSeries<ObservablePoint>
                {
                    Name = "WCSS",
                    Values = _elbowCurve.Select(p => new ObservablePoint(p.k, p.wcss)).ToArray(),
                    Stroke = new SolidColorPaint(SKColor.Parse("#087F73")) { StrokeThickness = 2.5f },
                    Fill = null, GeometrySize = 10, LineSmoothness = 0,
                    GeometryStroke = new SolidColorPaint(SKColor.Parse("#087F73")) { StrokeThickness = 2 },
                    GeometryFill = new SolidColorPaint(SKColors.White)
                }
            };
            var chosen = _elbowCurve.FirstOrDefault(p => p.k == k);
            if (chosen.k > 0)
                series.Add(new ScatterSeries<ObservablePoint>
                {
                    Name = $"Chosen k = {k}",
                    Values = new[] { new ObservablePoint(chosen.k, chosen.wcss) },
                    Fill = new SolidColorPaint(SKColor.Parse("#E5484D")), Stroke = null, GeometrySize = 16
                });
            ClustElbowChart.Series = series.ToArray();
            ClustElbowChart.XAxes = new[] { new Axis { Name = "Number of clusters (k)", MinStep = 1, NamePaint = new SolidColorPaint(SKColors.Gray), LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 } };
            ClustElbowChart.YAxes = new[] { new Axis { Name = "WCSS", NamePaint = new SolidColorPaint(SKColors.Gray), LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 } };
        }

        private void ElbowChart_DataPointerDown(IChartView chart, IEnumerable<ChartPoint> points)
        {
            var hit = points?.Select(p => p.Context.DataSource as ObservablePoint).FirstOrDefault(p => p?.X != null);
            if (hit == null) return;
            Dispatcher.Invoke(() => TxtClustK.Text = ((int)Math.Round(hit.X.Value)).ToString(CultureInfo.InvariantCulture));
        }

        // ---------------- Dendrogram (Hierarchical) ----------------
        private async void ClustDendrogram_Click(object sender, RoutedEventArgs e)
        {
            if (_dataset == null) { TxtClustResults.Text = "Load data first!"; return; }
            if (_dataset.RowCount > ClusteringEngine.MaxHierarchicalRows)
            { TxtClustResults.Text = $"Hierarchical clustering supports up to {ClusteringEngine.MaxHierarchicalRows} rows; this dataset has {_dataset.RowCount}."; return; }
            var data = _dataset;
            int resetVersion = _pageResetVersions[4];
            var engine = new ClusteringEngine { Type = ClustererType.Hierarchical, Linkage = SelectedLinkage() };
            SetClustBusy(true, "Building the dendrogram...");
            try
            {
                var tree = await Task.Run(() => engine.BuildDendrogram(data));
                if (resetVersion != _pageResetVersions[4] || !ReferenceEquals(data, _dataset)) return;
                _dendrogram = tree; _dendroDataset = data;
                ClustDendroBorder.Visibility = Visibility.Visible;
                TxtDendroInfo.Text = $"{tree.N} rows · {tree.Linkage} linkage. Click at a height to cut the tree; each branch crossing the red line becomes a cluster.";
                TxtClustResults.Text = "Dendrogram ready. Step 2: click the tree (or type the number of clusters), then click RUN CLUSTERING.";
                await Dispatcher.InvokeAsync(DrawDendrogram, System.Windows.Threading.DispatcherPriority.Loaded);
            }
            catch (Exception ex) { if (resetVersion == _pageResetVersions[4]) TxtClustResults.Text = "❌ Dendrogram: " + ex.Message; }
            finally { SetClustBusy(false); UpdateClustStepHint(); }
        }

        private const double DendroLeft = 52, DendroRight = 12, DendroTop = 14, DendroBottom = 26;

        private double DendroMaxHeight => _dendrogram == null || _dendrogram.Height.Length == 0 ? 1 : Math.Max(1e-12, _dendrogram.Height[^1]) * 1.05;

        private void DrawDendrogram()
        {
            var canvas = ClustDendroCanvas;
            canvas.Children.Clear();
            var tree = _dendrogram;
            if (tree == null || tree.N < 2) return;
            double w = canvas.ActualWidth > 50 ? canvas.ActualWidth : 600, h = canvas.ActualHeight > 50 ? canvas.ActualHeight : 360;
            double plotW = w - DendroLeft - DendroRight, plotH = h - DendroTop - DendroBottom, maxH = DendroMaxHeight;
            double Y(double height) => DendroTop + plotH * (1 - height / maxH);

            int n = tree.N, k = CurrentClustK();
            var xOf = new double[2 * n - 1];
            for (int i = 0; i < n; i++) xOf[tree.LeafOrder[i]] = DendroLeft + plotW * (i + 0.5) / n;
            var yOf = new double[2 * n - 1];
            for (int i = 0; i < n; i++) yOf[i] = Y(0);

            // Colour each branch below the cut by its cluster (same palette as the scatter).
            int[] labels = k > 0 ? tree.Cut(Math.Min(k, n)) : null;
            var nodeCluster = new int[2 * n - 1];
            for (int i = 0; i < n; i++) nodeCluster[i] = labels?[i] ?? -1;
            int cutMerges = k > 0 ? n - Math.Min(k, n) : 0;

            var segments = new Dictionary<string, List<(Point, Point)>>();
            void Seg(string hex, Point a, Point b) { if (!segments.TryGetValue(hex, out var list)) segments[hex] = list = new(); list.Add((a, b)); }

            for (int s = 0; s < tree.Height.Length; s++)
            {
                int l = tree.Left[s], r = tree.Right[s], node = n + s;
                xOf[node] = (xOf[l] + xOf[r]) / 2; yOf[node] = Y(tree.Height[s]);
                bool belowCut = s < cutMerges;
                nodeCluster[node] = belowCut ? nodeCluster[l] : -1;
                string hex = belowCut && nodeCluster[node] >= 0 ? ClustPalette[nodeCluster[node] % ClustPalette.Length] : "#64748B";
                string hexL = nodeCluster[l] >= 0 ? ClustPalette[nodeCluster[l] % ClustPalette.Length] : "#64748B";
                string hexR = nodeCluster[r] >= 0 ? ClustPalette[nodeCluster[r] % ClustPalette.Length] : "#64748B";
                Seg(hexL, new Point(xOf[l], yOf[l]), new Point(xOf[l], yOf[node]));
                Seg(hexR, new Point(xOf[r], yOf[r]), new Point(xOf[r], yOf[node]));
                Seg(hex, new Point(xOf[l], yOf[node]), new Point(xOf[r], yOf[node]));
            }
            foreach (var kv in segments)
            {
                var g = new StreamGeometry();
                using (var ctx = g.Open())
                    foreach (var (a, b) in kv.Value) { ctx.BeginFigure(a, false, false); ctx.LineTo(b, true, false); }
                g.Freeze();
                canvas.Children.Add(new Path { Data = g, Stroke = ClustB(kv.Key), StrokeThickness = n > 400 ? 0.8 : 1.4 });
            }

            // Height axis
            canvas.Children.Add(new Line { X1 = DendroLeft - 4, X2 = DendroLeft - 4, Y1 = DendroTop, Y2 = DendroTop + plotH, Stroke = Brushes.Gray, StrokeThickness = 1 });
            for (int t = 0; t <= 4; t++)
            {
                double v = maxH * t / 4, y = Y(v);
                canvas.Children.Add(new Line { X1 = DendroLeft - 8, X2 = DendroLeft - 4, Y1 = y, Y2 = y, Stroke = Brushes.Gray, StrokeThickness = 1 });
                var label = new TextBlock { Text = v.ToString("G3", CultureInfo.InvariantCulture), FontSize = 10, Foreground = Brushes.Gray };
                Canvas.SetLeft(label, 2); Canvas.SetTop(label, y - 7); canvas.Children.Add(label);
            }
            var axisName = new TextBlock { Text = n <= 60 ? "Rows (leaf labels = row index)" : $"{n} rows", FontSize = 10, Foreground = Brushes.Gray };
            Canvas.SetLeft(axisName, DendroLeft); Canvas.SetTop(axisName, h - 14); canvas.Children.Add(axisName);
            if (n <= 60)
                for (int i = 0; i < n; i++)
                {
                    var leaf = new TextBlock { Text = tree.LeafOrder[i].ToString(CultureInfo.InvariantCulture), FontSize = 8, Foreground = Brushes.Gray };
                    Canvas.SetLeft(leaf, xOf[tree.LeafOrder[i]] - 4); Canvas.SetTop(leaf, DendroTop + plotH + 1); canvas.Children.Add(leaf);
                }

            // Cut line for the chosen number of clusters
            if (k > 0 && k <= n)
            {
                double y = Y(tree.CutHeight(k));
                canvas.Children.Add(new Line { X1 = DendroLeft, X2 = w - DendroRight, Y1 = y, Y2 = y, Stroke = ClustB("#E5484D"), StrokeThickness = 1.6, StrokeDashArray = new DoubleCollection { 5, 3 } });
                var cut = new TextBlock { Text = $"k = {k}", FontSize = 11, FontWeight = FontWeights.Bold, Foreground = ClustB("#E5484D") };
                Canvas.SetLeft(cut, w - DendroRight - 44); Canvas.SetTop(cut, Math.Max(0, y - 16)); canvas.Children.Add(cut);
            }
        }

        private void ClustDendro_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (_dendrogram == null || _dendrogram.N < 2) return;
            double h = ClustDendroCanvas.ActualHeight, plotH = h - DendroTop - DendroBottom;
            double y = Math.Max(DendroTop, Math.Min(DendroTop + plotH, e.GetPosition(ClustDendroCanvas).Y));
            double height = (1 - (y - DendroTop) / plotH) * DendroMaxHeight;
            int k = Math.Max(1, _dendrogram.ClustersAtHeight(height));
            TxtClustK.Text = k.ToString(CultureInfo.InvariantCulture); // redraws the cut line
        }

        private void ClustDendro_SizeChanged(object sender, SizeChangedEventArgs e) { if (_dendrogram != null) DrawDendrogram(); }

        // ---------------- shared ----------------
        private void ClustK_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (ClustElbowBorder == null) return; // during InitializeComponent
            if (ClustElbowBorder.Visibility == Visibility.Visible) DrawElbow();
            if (ClustDendroBorder.Visibility == Visibility.Visible) DrawDendrogram();
        }

        private void ClustLinkage_Changed(object sender, SelectionChangedEventArgs e)
        {
            if (ClustDendroBorder == null) return;
            if (_dendrogram != null && _dendrogram.Linkage != SelectedLinkage())
            {
                TxtDendroInfo.Text = $"Linkage changed to {SelectedLinkage()}. Click ‘Show dendrogram’ again to rebuild the tree.";
                UpdateClustStepHint();
            }
        }

        private void SetClustBusy(bool busy, string message = null)
        {
            BtnClustRun.IsEnabled = BtnClustElbow.IsEnabled = BtnClustDendro.IsEnabled = !busy;
            ClustProgress.Visibility = busy ? Visibility.Visible : Visibility.Collapsed;
            ClustProgress.IsIndeterminate = busy;
            if (message != null) TxtClustResults.Text = message;
        }

        private void UpdateClustStepHint()
        {
            if (TxtClustStepHint == null) return;
            int algo = CmbClustAlgo.SelectedIndex;
            TxtClustStepHint.Text = algo == 0
                ? (ElbowReady ? "Step 2 · Choose k from the elbow chart (click a point) or type it, then run." : "Show the elbow method first; RUN CLUSTERING unlocks after it.")
                : algo == 2
                ? (DendrogramReady ? "Step 2 · Click the dendrogram at a height or type the number of clusters, then run." : "Show the dendrogram first; RUN CLUSTERING unlocks after it.")
                : "";
            TxtClustStepHint.Visibility = string.IsNullOrEmpty(TxtClustStepHint.Text) ? Visibility.Collapsed : Visibility.Visible;
            TxtClustKLabel.Text = algo == 0 ? "Step 2 · Clusters (k)" : algo == 2 ? "Step 2 · Number of clusters" : "Clusters (k)";
        }

        // Hides the elbow chart and dendrogram (new data, preprocessing change or page reset).
        private void ClearClusterDiagnostics()
        {
            _elbowCurve = null; _elbowDataset = null; _dendrogram = null; _dendroDataset = null;
            if (ClustElbowBorder == null) return;
            ClustElbowBorder.Visibility = ClustDendroBorder.Visibility = Visibility.Collapsed;
            ClustElbowChart.Series = Array.Empty<ISeries>();
            ClustDendroCanvas.Children.Clear();
            UpdateClustStepHint();
        }
    }
}
