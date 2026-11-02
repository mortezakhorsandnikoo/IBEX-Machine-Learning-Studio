using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using ArchiAI.ML;

// LiveCharts2
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using LiveChartsCore.SkiaSharpView.Painting.Effects;
using LiveChartsCore.Defaults;
using SkiaSharp;

namespace ArchiAI.Views
{
    // Second partial for the Polynomial Regression tab.
    // Shares _dataset / _loadedCsvPath with the main MLView partial.
    public partial class MLView
    {
        private PolynomialRegressionResult _polyResult;
        private bool _polyCrossValidated;
        public bool IsTrainingPolynomial { get; private set; }

        private void ResetPolynomialResults()
        {
            PolyCvTable.Show(null);
            _polyResult = null;
            _polyModel = null;
            _polyInputs.Clear();
            PolyPredictInputs.Children.Clear();
            PolyPredictResult.Text = "";
            PolyMetricsText.Text = "Train a model to see metrics.";
            PolyEquationText.Text = "";
            // Keep plot surfaces measured, even without data: collapsing them can skip the first render.
            PolyChartStatus.Text = "Ready to explore. Select a target and train your model to reveal the charts.";
            foreach (var chart in new[] { PolyCostChart, PolyScatterChart, PolyResidualChart, PolyImportanceChart })
                chart.Series = Array.Empty<ISeries>();
        }

        // ---- Theme colors (match the app) ----
        private static readonly SKColor PolyAccent = SKColor.Parse("#087F73");
        private static readonly SKColor PolyYellow = SKColor.Parse("#5275C5");
        private static readonly SKColor PolyRed = SKColor.Parse("#ff6b6b");
        private static readonly SKColor PolyText = SKColor.Parse("#334155");
        private static readonly SKColor PolyGrid = SKColor.Parse("#E2E7EF");

        // Call this from LoadCsv_Click / RefreshDataPreview so the target combo
        // always reflects the CURRENT (post-preprocessing) dataset.
        public void RefreshPolyCombos()
        {
            if (PolyTargetCombo == null) return;
            int prev = PolyTargetCombo.SelectedIndex;
            PolyTargetCombo.Items.Clear();
            if (_dataset != null)
                foreach (var h in _dataset.Headers)
                    PolyTargetCombo.Items.Add(h);
            if (prev >= 0 && prev < PolyTargetCombo.Items.Count)
                PolyTargetCombo.SelectedIndex = prev;
            else if (PolyTargetCombo.Items.Count > 0)
                PolyTargetCombo.SelectedIndex = PolyTargetCombo.Items.Count - 1;
        }

        // Fires whenever the Polynomial panel becomes visible (tab opened / algorithm
        // selected). Repopulates the target combo from the CURRENT preprocessed dataset,
        // so you never have to wire anything into RefreshDataPreview().
        private void PolyPanel_IsVisibleChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            if (e.NewValue is bool visible && visible) RefreshPolyCombos();
        }

        // Algorithm selector for the single "Regression" tab.
        // Shows exactly one algorithm's panel. To add a new regression algorithm later:
        //   1) add a <ComboBoxItem> to CmbRegAlgo in the XAML,
        //   2) add its panel Grid (x:Name="Panel_Xxx") to the tab,
        //   3) add one line here toggling that panel.
        private void RegAlgo_Changed(object sender, SelectionChangedEventArgs e)
        {
            if (PanelLinear == null || PanelPoly == null || PanelAdvanced == null) return;
            int idx = CmbRegAlgo.SelectedIndex;
            // 0 = Linear, 1 = Polynomial, 2+ = advanced IRegressionModel algorithms
            PanelLinear.Visibility = idx == 0 ? Visibility.Visible : Visibility.Collapsed;
            PanelPoly.Visibility = idx == 1 ? Visibility.Visible : Visibility.Collapsed;
            PanelAdvanced.Visibility = idx >= 2 ? Visibility.Visible : Visibility.Collapsed;
            if (idx >= 2) AdvOnSelected();   // populate target + build hyperparameter sliders
        }

        private void PolySplit_Changed(object sender, RoutedEventArgs e)
        {
            if (PolySplitLabel != null && PolySplitSlider != null)
                PolySplitLabel.Text = $"Train: {PolySplitSlider.Value:P0} / Test: {(1 - PolySplitSlider.Value):P0}";
        }

        private async void PolyTrain_Click(object sender, RoutedEventArgs e)
        {
            if (IsTrainingPolynomial) return;
            if (_dataset == null || _dataset.RowCount < 2)
            {
                PolyChartStatus.Text = "Import a CSV with at least two numeric rows from the Dataset page.";
                return;
            }
            if (PolyTargetCombo.SelectedIndex < 0)
            {
                PolyChartStatus.Text = "Select the target column you want to predict.";
                return;
            }

            int target = PolyTargetCombo.SelectedIndex;
            int degree = ParseInt(PolyDegreeBox?.Text, 2, 1, 8);
            double lr = ParseDouble(PolyLearningRateBox?.Text, 0.01);
            int iters = ParseInt(PolyIterationsBox?.Text, 1000, 1, 200000);
            double split = PolySplitSlider?.Value ?? 0.8;
            bool useCV = PolyUseCV?.IsChecked == true;
            int folds = ParseInt(PolyFoldsBox?.Text, 5, 2, 20);
            var dataset = _dataset;
            ResetPolynomialResults();
            IsTrainingPolynomial = true;
            IsEnabled = false;
            PolyChartStatus.Text = "Training polynomial model… Your results will appear here.";
            try
            {
                var pipeline = CapturePredictionRecipe(target);
                PolynomialCVResult cv = null;
                var result = await Task.Run(() =>
                {
                    if (useCV)
                    {
                        cv = PolynomialRegressionEngine.CrossValidate(dataset, target, degree, lr, iters, folds);
                        return cv.FinalModel;
                    }
                    return PolynomialRegressionEngine.Train(dataset, target, degree, lr, iters, split);
                });
                _polyResult = result;
                _polyCrossValidated = useCV;
                PolyDisplayResults(cv);
                // LiveCharts must measure a visible surface before accepting its first series.
                PolyChartGrid.Visibility = Visibility.Visible;
                UpdateLayout();
                await Dispatcher.InvokeAsync(() => { }, System.Windows.Threading.DispatcherPriority.Loaded);
                PolyDrawCostChart();
                PolyDrawScatter();
                PolyDrawResidual();
                PolyDrawImportance();
                foreach (var chart in new[] { PolyCostChart, PolyScatterChart, PolyResidualChart, PolyImportanceChart })
                    chart.CoreChart.Update(new LiveChartsCore.Kernel.ChartUpdateParams { Throttling = false });
                BuildPolyPredictInputs(pipeline);
                PolyScatterCaption.Text = useCV ? "Teal: all-data fit · Dashed: ideal" : "Teal: train · Blue: test · Dashed: ideal";
                PolyChartStatus.Text = useCV
                    ? $"{cv.K}-fold validation complete. Metrics summarize held-out folds; charts show the final fit on all data."
                    : $"Training complete · {result.TrainActual.Length} training rows · {result.TestActual.Length} test rows. Scroll to inspect all four charts.";
                MLRunHistory.AddRegression("Polynomial Regression", result.TargetName,
                    cv?.MeanR2 ?? result.R2, double.NaN, cv?.MeanMAE ?? result.MAE, cv?.MeanRMSE ?? result.RMSE);
            }
            catch (Exception ex)
            {
                ResetPolynomialResults();
                PolyChartStatus.Text = "Could not train: " + ex.Message;
            }
            finally
            {
                IsTrainingPolynomial = false;
                IsEnabled = true;
            }
        }

        private void PolyDisplayResults(PolynomialCVResult cv = null)
        {
            PolyCvTable.Show(cv?.Report);
            var r = _polyResult;
            if (PolyMetricsText != null)
            {
                if (cv != null)
                    PolyMetricsText.Text =
                        $"Degree: {r.Degree}\n" +
                        $"{cv.K}-Fold CV\n" +
                        $"R²:   {cv.MeanR2:0.####} ± {cv.StdR2:0.####}\n" +
                        $"RMSE: {cv.MeanRMSE:0.####} ± {cv.StdRMSE:0.####}\n" +
                        $"MAE:  {cv.MeanMAE:0.####} ± {cv.StdMAE:0.####}\n" +
                        $"Full-data R²: {r.TrainR2:0.####}";
                else
                    PolyMetricsText.Text =
                        $"Degree: {r.Degree}\n" +
                        $"R² (test): {r.R2:0.####}\n" +
                        $"R² (train): {r.TrainR2:0.####}\n" +
                        $"MSE: {r.MSE:0.####}\n" +
                        $"RMSE: {r.RMSE:0.####}\n" +
                        $"MAE: {r.MAE:0.####}";
            }
            if (PolyEquationText != null)
                PolyEquationText.Text = r.Equation;
        }

        private void PolyDrawCostChart()
        {
            var pts = _polyResult.CostHistory
                .Select((c, i) => new ObservablePoint(i, c)).ToArray();

            PolyCostChart.Series = new ISeries[]
            {
                new LineSeries<ObservablePoint>
                {
                    Values = pts,
                    Name = "Cost",
                    Stroke = new SolidColorPaint(PolyAccent, 2),
                    Fill = null,
                    GeometrySize = 0,
                    LineSmoothness = 0
                }
            };
            PolyCostChart.XAxes = MakeAxes("Iteration");
            PolyCostChart.YAxes = MakeAxes("Cost");
        }

        private void PolyDrawScatter()
        {
            var train = _polyResult.TrainActual
                .Select((a, i) => new ObservablePoint(a, _polyResult.TrainPredictions[i])).ToArray();
            var test = (_polyCrossValidated ? Array.Empty<double>() : _polyResult.TestActual)
                .Select((a, i) => new ObservablePoint(a, _polyResult.TestPredictions[i])).ToArray();

            // Ideal y = x reference line spanning the actual range.
            var allActual = _polyResult.TrainActual.Concat(_polyResult.TestActual).ToArray();
            double lo = allActual.Min(), hi = allActual.Max();
            if (lo == hi) { lo -= 0.5; hi += 0.5; }

            PolyScatterChart.Series = new ISeries[]
            {
                new LineSeries<ObservablePoint>
                {
                    Values = new [] { new ObservablePoint(lo, lo), new ObservablePoint(hi, hi) },
                    Name = "Ideal",
                    Stroke = new SolidColorPaint(PolyText, 1) { PathEffect = new DashEffect(new float[]{4,4}) },
                    Fill = null, GeometrySize = 0
                },
                new ScatterSeries<ObservablePoint>
                {
                    Values = train, Name = _polyCrossValidated ? "All-data fit" : "Train",
                    Fill = new SolidColorPaint(PolyAccent), GeometrySize = 7
                },
                new ScatterSeries<ObservablePoint>
                {
                    Values = test, Name = "Test",
                    Fill = new SolidColorPaint(PolyYellow), GeometrySize = 7
                }
            };
            PolyScatterChart.XAxes = MakeAxes("Actual");
            PolyScatterChart.YAxes = MakeAxes("Predicted");
        }

        private void PolyDrawResidual()
        {
            var res = _polyResult.TestActual
                .Select((a, i) => new ObservablePoint(_polyResult.TestPredictions[i], a - _polyResult.TestPredictions[i]))
                .ToArray();

            PolyResidualChart.Series = new ISeries[]
            {
                new ScatterSeries<ObservablePoint>
                {
                    Values = res, Name = "Residual",
                    Fill = new SolidColorPaint(PolyRed), GeometrySize = 7
                }
            };
            PolyResidualChart.XAxes = MakeAxes("Predicted");
            PolyResidualChart.YAxes = MakeAxes("Residual");
        }

        private void PolyDrawImportance()
        {
            var names = _polyResult.FeatureNames;
            var vals = _polyResult.Weights.Select(Math.Abs).ToArray();

            PolyImportanceChart.Series = new ISeries[]
            {
                new ColumnSeries<double>
                {
                    Values = vals, Name = "|Weight|",
                    Fill = new SolidColorPaint(PolyAccent)
                }
            };
            PolyImportanceChart.XAxes = new Axis[]
            {
                new Axis
                {
                    Labels = names.ToList(),
                    LabelsRotation = 45,
                    LabelsPaint = new SolidColorPaint(PolyText),
                    SeparatorsPaint = new SolidColorPaint(PolyGrid)
                }
            };
            PolyImportanceChart.YAxes = MakeAxes("Importance");
        }

        // ---- shared small helpers (also used by Visualization partial) ----
        // Returns Axis[]; IEnumerable<Axis> is covariantly assignable to the
        // chart's XAxes/YAxes (IEnumerable<ICartesianAxis>), so no extra using needed.
        private static Axis[] MakeAxes(string name) => new Axis[]
        {
            new Axis
            {
                Name = name,
                TextSize = 11,
                NameTextSize = 12,
                NamePaint = new SolidColorPaint(PolyText),
                LabelsPaint = new SolidColorPaint(PolyText),
                SeparatorsPaint = new SolidColorPaint(PolyGrid)
            }
        };

        private static int ParseInt(string s, int fallback, int min, int max)
        {
            if (int.TryParse(s, out int v)) return Math.Max(min, Math.Min(max, v));
            return fallback;
        }

        private static double ParseDouble(string s, double fallback)
        {
            if (double.TryParse(s, out double v)) return v;
            return fallback;
        }
    }
}
