using ArchiAI.ML;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.WPF;
using System;
using System.Linq;
using System.Windows;
using System.Windows.Controls;

namespace ArchiAI.Views
{
    public partial class MLView
    {
        private readonly int[] _pageResetVersions = new int[7];

        private void ResetPage_Click(object sender, RoutedEventArgs e)
        {
            ResetWorkspacePage(MainTabs.SelectedIndex);
            ExportStatus.Text = "";
        }

        private void ResetWorkspacePage(int page)
        {
            if (page < 0 || page >= 7) return;
            _pageResetVersions[page]++;
            switch (page)
            {
                case 0:
                    TxtFileName.Text = "Import a CSV to get started.";
                    TxtDataInfo.Text = TxtDataStatus.Text = TxtColStats.Text = "";
                    CmbStatCol.Items.Clear();
                    break;
                case 1:
                    foreach (var border in new[] { DescribeBorder, MissingBorder, MissingChartBorder, AnomalyBorder,
                        AnomalyChartBorder, NormalizeBorder, NormChartBorder, EncoderBorder, EncoderPreviewBorder,
                        PcaBorder, PcaScreeBorder, PcaCumBorder, PcaScatterBorder }) border.Visibility = Visibility.Collapsed;
                    TxtMissingReport.Text = TxtAnomalyReport.Text = TxtEncoderReport.Text = TxtPcaReport.Text = "";
                    TxtNormTitle.Text = "Normalized Data";
                    TxtPrepStatus.Text = "";
                    break;
                case 2:
                    _result = null; _linModel = null; _linCv = null;
                    _linInputs.Clear(); LinPredictInputs.Children.Clear();
                    TxtResults.Text = TxtWeights.Text = LinPredictResult.Text = "";
                    LinCvTable.Show(null);
                    ResetPolynomialResults(); PolyCvTable.Show(null);
                    PolyMetricsText.Text = PolyChartStatus.Text = "";
                    _advModel?.Dispose(); _advModel = null; _advPipeline = null; _advFeatureNames = null;
                    _advPredictBoxes.Clear(); AdvPredictInputs.Children.Clear();
                    TxtAdvResults.Text = AdvPredictResult.Text = "";
                    AdvCvTable.Show(null); AdvTreeBorder.Visibility = Visibility.Collapsed;
                    AdvProgress.Visibility = Visibility.Collapsed; BtnAdvTrain.IsEnabled = true;
                    break;
                case 3:
                    _clsResult = null; _clsCv = null;
                    _clsPredictBoxes.Clear(); ClsPredictInputs.Children.Clear();
                    TxtClsResults.Text = ClsPredictResult.Text = ""; ClsCvTable.Show(null);
                    break;
                case 4:
                    _clustResult = null; _clustPipeline = null;
                    _clustPredictBoxes.Clear(); ClustPredictInputs.Children.Clear();
                    TxtClustResults.Text = ClustPredictResult.Text = "";
                    ClustProgress.Visibility = Visibility.Collapsed; BtnClustRun.IsEnabled = true;
                    break;
                case 5:
                    MLRunHistory.Clear(); TxtCompareInfo.Text = "";
                    break;
                case 6:
                    ResetVisualization(); VizStatus.Text = "";
                    break;
            }
            foreach (var (_, grid) in TablesForTab(page)) grid.ItemsSource = null;
            foreach (var (_, element) in ChartsForTab(page))
                switch (element)
                {
                    case CartesianChart chart:
                        chart.Series = Array.Empty<ISeries>();
                        chart.XAxes = new[] { new Axis() }; chart.YAxes = new[] { new Axis() }; break;
                    case PieChart chart: chart.Series = Array.Empty<ISeries>(); break;
                    case PolarChart chart: chart.Series = Array.Empty<ISeries>(); break;
                    case Canvas canvas: canvas.Children.Clear(); break;
                }
        }
    }
}
