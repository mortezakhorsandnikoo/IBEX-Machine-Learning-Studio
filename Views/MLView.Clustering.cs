using ArchiAI.ML;
using LiveChartsCore;
using LiveChartsCore.Defaults;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace ArchiAI.Views
{
    public partial class MLView
    {
        private ClusterResult _clustResult;
        private readonly List<TextBox> _clustPredictBoxes = new();
        private static readonly string[] ClustPalette =
            { "#087F73", "#ff6b6b", "#5275C5", "#6ba3ff", "#c792ea", "#7bed9f", "#ff9f43", "#f368e0", "#48dbfb", "#feca57" };

        private void ClustPanel_IsVisibleChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            if (!(bool)e.NewValue) return;
            if (_clustPredictBoxes.Count == 0 && _clustResult != null) BuildClustPredictInputs();
        }

        private void ClustAlgo_Changed(object sender, SelectionChangedEventArgs e)
        {
            if (ClustKParams == null) return;
            int i = CmbClustAlgo.SelectedIndex; // 0 KMeans,1 DBSCAN,2 Hierarchical,3 Fuzzy
            ClustKParams.Visibility = (i == 0 || i == 2 || i == 3) ? Visibility.Visible : Visibility.Collapsed;
            ClustIterParams.Visibility = (i == 0 || i == 3) ? Visibility.Visible : Visibility.Collapsed;
            ClustFuzzyParams.Visibility = i == 3 ? Visibility.Visible : Visibility.Collapsed;
            ClustDbscanParams.Visibility = i == 1 ? Visibility.Visible : Visibility.Collapsed;
            ClustLinkParams.Visibility = i == 2 ? Visibility.Visible : Visibility.Collapsed;
        }

        private async void ClustRun_Click(object sender, RoutedEventArgs e)
        {
            if (_dataset == null) { TxtClustResults.Text = "Load data first!"; return; }
            var engine = new ClusteringEngine { Type = (ClustererType)CmbClustAlgo.SelectedIndex };
            if (int.TryParse(TxtClustK.Text, out int k) && k > 0) engine.K = k;
            if (int.TryParse(TxtClustIter.Text, out int it) && it > 0) engine.Iterations = it;
            if (double.TryParse(TxtClustM.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out double mm) && mm > 1) engine.Fuzziness = mm;
            if (double.TryParse(TxtClustEps.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out double eps) && eps > 0) engine.Eps = eps;
            if (int.TryParse(TxtClustMinPts.Text, out int mp) && mp > 0) engine.MinPts = mp;
            engine.Linkage = (CmbClustLinkage.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "average";

            BtnClustRun.IsEnabled = false; ClustProgress.Visibility = Visibility.Visible; ClustProgress.IsIndeterminate = true;
            TxtClustResults.Text = "Clustering...";
            int resetVersion = _pageResetVersions[4];
            var trainingData = _dataset;
            try
            {
                var pipeline = CapturePredictionRecipe(-1);
                var result = await Task.Run(() => engine.Run(trainingData));
                if (resetVersion != _pageResetVersions[4]) return;
                _clustResult = result;
                _clustPipeline = pipeline;
                DisplayClust();
                BuildClustPredictInputs();
            }
            catch (Exception ex) { if (resetVersion == _pageResetVersions[4]) TxtClustResults.Text = "❌ " + ex.Message; }
            ClustProgress.Visibility = Visibility.Collapsed; BtnClustRun.IsEnabled = true;
        }

        private void DisplayClust()
        {
            var r = _clustResult; if (r == null) return;
            var sb = new StringBuilder();
            sb.AppendLine(r.Info);
            sb.AppendLine($"Clusters:    {r.NumClusters}");
            sb.AppendLine($"Silhouette:  {r.Silhouette:F4}   (−1..1, higher = better)");
            sb.AppendLine();
            sb.AppendLine("Cluster sizes:");
            for (int c = 0; c < r.ClusterSizes.Length; c++) sb.AppendLine($"  Cluster {c}: {r.ClusterSizes[c]}");
            if (r.Labels.Any(l => l < 0)) sb.AppendLine($"  Noise: {r.Labels.Count(l => l < 0)}");
            TxtClustResults.Text = sb.ToString();

            // scatter: one series per cluster (+ noise), positioned by PC1/PC2
            var series = new List<ISeries>();
            var byLabel = new Dictionary<int, List<ObservablePoint>>();
            for (int i = 0; i < r.Labels.Length; i++)
            {
                int l = r.Labels[i];
                if (!byLabel.ContainsKey(l)) byLabel[l] = new List<ObservablePoint>();
                byLabel[l].Add(new ObservablePoint(r.Proj2D[i][0], r.Proj2D[i][1]));
            }
            foreach (var kv in byLabel.OrderBy(k => k.Key))
            {
                string hex = kv.Key < 0 ? "#A8A29B" : ClustPalette[kv.Key % ClustPalette.Length];
                series.Add(new ScatterSeries<ObservablePoint>
                {
                    Name = kv.Key < 0 ? "noise" : $"C{kv.Key}",
                    Values = kv.Value,
                    Fill = new SolidColorPaint(SKColor.Parse(hex)),
                    GeometrySize = 6, Stroke = null
                });
            }
            ClustScatter.Series = series.ToArray();
            ClustScatter.XAxes = new Axis[] { new Axis { Name = "PC1", NamePaint = new SolidColorPaint(SKColors.Gray), LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 } };
            ClustScatter.YAxes = new Axis[] { new Axis { Name = "PC2", NamePaint = new SolidColorPaint(SKColors.Gray), LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 } };
        }

        private void BuildClustPredictInputs()
        {
            ClustPredictInputs.Children.Clear();
            _clustPredictBoxes.Clear();
            if (_dataset == null) return;
            foreach (var name in _clustPipeline?.InputNames ?? _dataset.Headers)
            {
                ClustPredictInputs.Children.Add(new TextBlock { Text = name, Foreground = ClustB("#334155"), FontSize = 12, Margin = new Thickness(0, 6, 0, 2) });
                var tb = new TextBox { Text = "0", FontSize = 13, Background = ClustB("#F1F5F9"), Foreground = Brushes.DarkSlateGray, BorderBrush = ClustB("#E2E7EF"), Padding = new Thickness(6, 4, 6, 4) };
                _clustPredictBoxes.Add(tb);
                ClustPredictInputs.Children.Add(tb);
            }
        }

        private void ClustPredict_Click(object sender, RoutedEventArgs e)
        {
            if (_clustResult == null) { ClustPredictResult.Text = "Run clustering first."; return; }
            var x = new double[_clustPredictBoxes.Count];
            for (int i = 0; i < x.Length; i++)
                if (!double.TryParse(_clustPredictBoxes[i].Text.Trim(), NumberStyles.Any, CultureInfo.InvariantCulture, out x[i]))
                { ClustPredictResult.Text = $"Invalid value for {(_clustPipeline?.InputNames ?? _dataset.Headers)[i]}"; return; }
            try { ClustPredictResult.Text = $"Assigned to cluster: {_clustResult.Assign(_clustPipeline?.Transform(x) ?? x)}"; }
            catch (Exception ex) { ClustPredictResult.Text = "Assign error: " + ex.Message; }
        }

        private static SolidColorBrush ClustB(string hex) => new SolidColorBrush((Color)ColorConverter.ConvertFromString(hex));
    }
}
