using ArchiAI.Helpers;
using ArchiAI.ML;
using LiveChartsCore;
using LiveChartsCore.Defaults;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using LiveChartsCore.SkiaSharpView.WPF;
using Microsoft.Win32;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Data;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace ArchiAI.Views
{
    public partial class MLView : Page
    {
        private MLDataset _dataset;
        private LinearRegressionResult _result;
        private LinearCVResult _linCv;

        public MLView()
        {
            InitializeComponent();
            InitChartDefaults();
        }

        private void InitChartDefaults()
        {
            var emptyAxes = new Axis[]
            {
                new Axis { LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 11 }
            };
            CostChart.XAxes = emptyAxes;
            CostChart.YAxes = emptyAxes;
        }


        // ==================== LABEL ENCODER ====================
        public class LabelEncoderResult
        {
            public string[] Headers { get; set; }
            public string[][] EncodedData { get; set; }
            public Dictionary<string, Dictionary<string, int>> Mappings { get; set; } // col -> (label -> int)
            public int[] EncodedColumnIndices { get; set; }
            public int TotalEncoded { get; set; }
        }
        // ===================== DATA TAB =====================

        private void LoadCsv_Click(object sender, RoutedEventArgs e)
        {
            if (IsTrainingPolynomial) return;
            var dlg = new OpenFileDialog { Filter = "CSV Files|*.csv" };
            if (dlg.ShowDialog() != true) return;

            LoadDataset(dlg.FileName);
        }

        public void ImportCsv() => LoadCsv_Click(this, new RoutedEventArgs());
        public void SelectSection(int index) => MainTabs.SelectedIndex = index;
        public void SelectPolynomial() => CmbRegAlgo.SelectedIndex = 1;

        public bool LoadDataset(string path)
        {
            if (IsTrainingPolynomial) return false;
            try
            {
                // Validate before replacing an existing workspace.
                var loaded = MLDataset.LoadCsv(path);
                if (loaded.RowCount < 2 || loaded.ColCount < 2)
                    throw new ArgumentException("Use a CSV with at least two numeric rows and two columns.");
                // ===== RESET EVERYTHING =====
                ResetAllTabs();

                // Load new data
                _dataset = loaded;
                _loadedCsvPath = path;
                TxtFileName.Text = System.IO.Path.GetFileName(path);

                // Fill DataGrid
                var dt = new DataTable();
                foreach (var h in _dataset.Headers)
                    dt.Columns.Add(h);
                int previewCount = Math.Min(_dataset.RowCount, 500);
                for (int i = 0; i < previewCount; i++)
                {
                    var row = dt.NewRow();
                    for (int j = 0; j < _dataset.ColCount; j++)
                        row[j] = _dataset.Rows[i][j].ToString("G6", CultureInfo.InvariantCulture);
                    dt.Rows.Add(row);
                }
                DataPreview.ItemsSource = dt.DefaultView;

                // Dataset info
                var sb = new StringBuilder();
                sb.AppendLine($"Rows:       {_dataset.RowCount}");
                sb.AppendLine($"Columns:    {_dataset.ColCount}");
                sb.AppendLine($"Features:   {_dataset.ColCount - 1}");
                sb.AppendLine();
                sb.AppendLine("Columns:");
                for (int i = 0; i < _dataset.ColCount; i++)
                    sb.AppendLine($"  [{i}] {_dataset.Headers[i]}");
                TxtDataInfo.Text = sb.ToString();

                // Fill combo boxes
                CmbStatCol.Items.Clear();
                CmbTarget.Items.Clear();
                foreach (var h in _dataset.Headers)
                {
                    CmbStatCol.Items.Add(h);
                    CmbTarget.Items.Add(h);
                }
                CmbStatCol.SelectedIndex = 0;
                CmbTarget.SelectedIndex = _dataset.ColCount - 1;

                // Correlation matrix
                DrawCorrelationMatrix();

                RefreshPolyCombos();
                RefreshVizCombos();
                TxtDataStatus.Text = $"Loaded {_dataset.RowCount} rows × {_dataset.ColCount} columns. Choose Prepare data or a modeling task to continue.";
                return true;
            }
            catch (Exception ex)
            {
                TxtDataStatus.Text = "Could not import dataset: " + ex.Message;
                return false;
            }
        }


        // ================ LABEL ENCODER ================
        private void LabelEncode_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_loadedCsvPath) || !File.Exists(_loadedCsvPath))
            {
                PrepStatus("⚠ Load a CSV file first from the Data tab!");
                return;
            }

            try
            {
                // Parse user-specified columns (if any)
                int[] targetCols = null;
                string colText = TxtEncoderCols.Text.Trim();
                if (!string.IsNullOrEmpty(colText))
                {
                    targetCols = colText.Split(',')
                        .Select(s => s.Trim())
                        .Where(s => int.TryParse(s, out _))
                        .Select(s => int.Parse(s))
                        .ToArray();

                    if (targetCols.Length == 0) targetCols = null;
                }

                var result = PreprocessingEngine.LabelEncode(_loadedCsvPath, targetCols);

                if (result == null)
                {
                    PrepStatus("No data to encode.");
                    return;
                }

                if (result.Mappings.Count == 0)
                {
                    PrepStatus("✅ No categorical columns detected. All columns are numeric.");
                    return;
                }

                // Build report
                var sb = new StringBuilder();
                sb.AppendLine($"Encoded Columns:    {result.Mappings.Count}");
                sb.AppendLine($"Total Values Encoded: {result.TotalEncoded}");
                sb.AppendLine();

                foreach (var kvp in result.Mappings)
                {
                    sb.AppendLine($"Column: {kvp.Key}");
                    sb.AppendLine("─────────────────────────────");
                    foreach (var mapping in kvp.Value.OrderBy(m => m.Value))
                    {
                        sb.AppendLine($"  {mapping.Key,-20} → {mapping.Value}");
                    }
                    sb.AppendLine();
                }

                TxtEncoderReport.Text = sb.ToString();
                EncoderBorder.Visibility = Visibility.Visible;

                // Preview table
                var dt = new DataTable();
                foreach (var h in result.Headers) dt.Columns.Add(h);
                int preview = Math.Min(result.EncodedData.Length, 200);
                for (int i = 0; i < preview; i++)
                {
                    var row = dt.NewRow();
                    for (int j = 0; j < result.Headers.Length; j++)
                        row[j] = result.EncodedData[i][j];
                    dt.Rows.Add(row);
                }
                EncoderGrid.ItemsSource = dt.DefaultView;
                EncoderPreviewBorder.Visibility = Visibility.Visible;

                // Save encoded file
                string dir = Path.GetDirectoryName(_loadedCsvPath);
                string name = Path.GetFileNameWithoutExtension(_loadedCsvPath);
                string encodedPath = Path.Combine(dir, name + "_encoded.csv");

                using (var w = new StreamWriter(encodedPath))
                {
                    w.WriteLine(string.Join(",", result.Headers));
                    foreach (var row in result.EncodedData)
                        w.WriteLine(string.Join(",", row));
                }

                // Reload dataset
                _dataset = MLDataset.LoadCsv(encodedPath);
                _loadedCsvPath = encodedPath;
                RefreshDataPreview();

                PrepStatus($"✅ Encoded {result.Mappings.Count} columns. Saved: {encodedPath}");
            }
            catch (Exception ex) { PrepStatus("Encoder error: " + ex.Message); }
        }

        private void StatCol_Changed(object sender, SelectionChangedEventArgs e)
        {
            if (_dataset == null || CmbStatCol.SelectedIndex < 0) return;
            int col = CmbStatCol.SelectedIndex;

            try
            {
                var sb = new StringBuilder();
                sb.AppendLine($"Column: {_dataset.Headers[col]}");
                sb.AppendLine($"────────────────────────");
                sb.AppendLine($"Count:      {_dataset.RowCount}");
                sb.AppendLine($"Missing:    {_dataset.MissingCount(col)}");
                sb.AppendLine($"Mean:       {_dataset.Mean(col):G6}");
                sb.AppendLine($"Median:     {_dataset.Median(col):G6}");
                sb.AppendLine($"Mode:       {_dataset.Mode(col):G6}");
                sb.AppendLine($"Std Dev:    {_dataset.StdDev(col):G6}");
                sb.AppendLine($"Min:        {_dataset.Min(col):G6}");
                sb.AppendLine($"Max:        {_dataset.Max(col):G6}");
                sb.AppendLine($"Range:      {_dataset.Range(col):G6}");
                sb.AppendLine($"Q1 (25%):   {_dataset.Q1(col):G6}");
                sb.AppendLine($"Q3 (75%):   {_dataset.Q3(col):G6}");
                sb.AppendLine($"IQR:        {_dataset.IQR(col):G6}");
                sb.AppendLine($"Skewness:   {_dataset.Skewness(col):G6}");
                sb.AppendLine($"Kurtosis:   {_dataset.Kurtosis(col):G6}");
                TxtColStats.Text = sb.ToString();

                // Histogram
                var (edges, counts) = _dataset.Histogram(col, 20);
                var values = new List<double>();
                var labels = new List<string>();
                for (int i = 0; i < counts.Length; i++)
                {
                    values.Add(counts[i]);
                    labels.Add(edges[i].ToString("G3"));
                }

                HistChart.Series = new ISeries[]
                {
                    new ColumnSeries<double>
                    {
                        Values = values,
                        Fill = new SolidColorPaint(SKColor.Parse("#087F73")),
                        Stroke = null,
                        MaxBarWidth = 50
                    }
                };
                HistChart.XAxes = new Axis[]
                {
                    new Axis
                    {
                        Labels = labels,
                        LabelsPaint = new SolidColorPaint(SKColors.Gray),
                        TextSize = 9,
                        LabelsRotation = 45
                    }
                };
                HistChart.YAxes = new Axis[]
                {
                    new Axis
                    {
                        LabelsPaint = new SolidColorPaint(SKColors.Gray),
                        TextSize = 10
                    }
                };
            }
            catch (Exception ex)
            {
                TxtColStats.Text = "Error: " + ex.Message;
            }
        }

        private void DrawCorrelationMatrix()
        {
            if (_dataset == null || DataPreview.ItemsSource == null) return;

            CorrCanvas.Children.Clear();
            int n = _dataset.ColCount;
            var corr = _dataset.CorrelationMatrix();

            double canvasW = CorrCanvas.ActualWidth > 10 ? CorrCanvas.ActualWidth : 340;
            double canvasH = Math.Max(200, n * 30 + 40);
            CorrCanvas.Height = canvasH;

            double labelW = 80;
            const double legendWidth = 68;
            double cellW = (canvasW - labelW - legendWidth) / n;
            double cellH = (canvasH - 30) / n;
            cellW = Math.Min(cellW, cellH);
            cellH = cellW;

            var dv = new DrawingVisual();
            using (var dc = dv.RenderOpen())
            {
                // Header labels
                for (int j = 0; j < n; j++)
                {
                    string name = _dataset.Headers[j].Length > 8
                        ? _dataset.Headers[j].Substring(0, 8) : _dataset.Headers[j];
                    var ft = MkTxt(name, 9, Brushes.Gray);
                    dc.DrawText(ft, new Point(labelW + j * cellW + cellW / 2 - ft.Width / 2, 0));
                }

                // Rows
                for (int i = 0; i < n; i++)
                {
                    // Row label
                    string name = _dataset.Headers[i].Length > 10
                        ? _dataset.Headers[i].Substring(0, 10) : _dataset.Headers[i];
                    var ft = MkTxt(name, 9, Brushes.Gray);
                    dc.DrawText(ft, new Point(0, 25 + i * cellH + cellH / 2 - ft.Height / 2));

                    // Cells
                    for (int j = 0; j < n; j++)
                    {
                        double val = corr[i, j];
                        Color c = CorrColor(val);
                        var rect = new Rect(labelW + j * cellW, 25 + i * cellH, cellW - 1, cellH - 1);
                        dc.DrawRectangle(new SolidColorBrush(c), null, rect);

                        // Value text
                        var vtxt = MkTxt(val.ToString("F2"), Math.Clamp(cellW * 0.28, 9, 16),
                            new SolidColorBrush(CorrelationTextColor(c)));
                        dc.DrawText(vtxt, new Point(
                            rect.X + rect.Width / 2 - vtxt.Width / 2,
                            rect.Y + rect.Height / 2 - vtxt.Height / 2));
                    }
                }
                // Fixed Pearson correlation scale, shared by every dataset and exported with the matrix.
                double legendX = labelW + n * cellW + 12;
                double legendHeight = n * cellH - 1;
                var ribbon = new LinearGradientBrush
                {
                    StartPoint = new Point(0, 0), EndPoint = new Point(0, 1),
                    ColorInterpolationMode = ColorInterpolationMode.SRgbLinearInterpolation
                };
                ribbon.GradientStops.Add(new GradientStop(CorrColor(1), 0));
                ribbon.GradientStops.Add(new GradientStop(CorrColor(0), 0.5));
                ribbon.GradientStops.Add(new GradientStop(CorrColor(-1), 1));
                dc.DrawRectangle(ribbon, new Pen(Brushes.LightGray, 0.5), new Rect(legendX, 25, 12, legendHeight));
                dc.DrawText(MkTxt("r", 10, Brushes.DimGray), new Point(legendX + 3, 0));
                dc.DrawText(MkTxt("+1 max", 9, Brushes.Black), new Point(legendX + 17, 25));
                var zeroLabel = MkTxt("0", 9, Brushes.Black);
                dc.DrawText(zeroLabel, new Point(legendX + 17, 25 + legendHeight / 2 - zeroLabel.Height / 2));
                var minLabel = MkTxt("−1 min", 9, Brushes.Black);
                dc.DrawText(minLabel, new Point(legendX + 17, 25 + legendHeight - minLabel.Height));
            }

            CorrCanvas.Children.Add(new DrawingVisualHost(dv));
        }

        private void CorrCanvas_SizeChanged(object sender, SizeChangedEventArgs e) => DrawCorrelationMatrix();

        internal static Color CorrelationTextColor(Color background)
        {
            // Select the higher-contrast text color from the actual cell luminance.
            double Linear(byte channel)
            {
                double v = channel / 255.0;
                return v <= 0.04045 ? v / 12.92 : Math.Pow((v + 0.055) / 1.055, 2.4);
            }
            double luminance = 0.2126 * Linear(background.R) + 0.7152 * Linear(background.G) + 0.0722 * Linear(background.B);
            return luminance > 0.179 ? Colors.Black : Colors.White;
        }

        private static Color CorrColor(double val)
        {
            // Blue (-1) → White (0) → Red (+1)
            val = Math.Max(-1, Math.Min(1, val));
            if (val >= 0)
            {
                byte r = 255;
                byte g = (byte)(255 * (1 - val));
                byte b = (byte)(255 * (1 - val));
                return Color.FromRgb(r, g, b);
            }
            else
            {
                double v = -val;
                byte r = (byte)(255 * (1 - v));
                byte g = (byte)(255 * (1 - v));
                byte b = 255;
                return Color.FromRgb(r, g, b);
            }
        }

        private static int ParseFolds(string s)
        {
            if (int.TryParse(s, out int k) && k >= 2) return Math.Min(k, 20);
            return 5;
        }

        // ===================== LINEAR REGRESSION TAB =====================

        private void Split_Changed(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (TxtSplit != null)
            {
                int train = (int)SliderSplit.Value;
                TxtSplit.Text = $"{train}/{100 - train}";
            }
        }

        private async void Train_Click(object sender, RoutedEventArgs e)
        {
            if (_dataset == null) { TxtResults.Text = "Load data first!"; return; }
            if (CmbTarget.SelectedIndex < 0) { TxtResults.Text = "Select target column!"; return; }

            int targetCol = CmbTarget.SelectedIndex;

            if (!int.TryParse(TxtIter.Text, out int iterations) || iterations < 1)
            { TxtResults.Text = "Invalid iterations."; return; }

            if (!double.TryParse(TxtLR.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out double lr) || lr <= 0)
            { TxtResults.Text = "Invalid learning rate."; return; }

            var regType = (RegularizationType)CmbReg.SelectedIndex;
            bool normalize= false;
            double split = SliderSplit.Value / 100.0;

            // Feature names (all except target)
            var featureNames = new List<string>();
            for (int i = 0; i < _dataset.ColCount; i++)
                if (i != targetCol) featureNames.Add(_dataset.Headers[i]);

            LinCvTable.Show(null);
            TxtResults.Text = "Training...";
            TxtWeights.Text = "";

            var engine = new LinearRegressionEngine
            {
                Iterations = iterations,
                LearningRate = lr,
                Regularization = regType,
                Normalize = normalize,
                TrainTestSplit = split
            };

       

        bool useCv = ChkKFold.IsChecked == true;
            int folds = ParseFolds(TxtFolds.Text);
            int resetVersion = _pageResetVersions[2];
            var trainingData = _dataset;
            try
            {
                var pipeline = CapturePredictionRecipe(targetCol);
                if (useCv)
                {
                    var cv = await Task.Run(() => engine.CrossValidate(trainingData, targetCol, featureNames.ToArray(), folds));
                    if (resetVersion != _pageResetVersions[2]) return;
                    _linCv = cv;
                    _result = cv.FinalModel;
                }
                else
                {
                    _linCv = null;
                    var result = await Task.Run(() => engine.Run(trainingData, targetCol, featureNames.ToArray()));
                    if (resetVersion != _pageResetVersions[2]) return;
                    _result = result;
                }
                DisplayResults();
                BuildLinPredictInputs(pipeline);
                try
                {
                    string tgt = CmbTarget.SelectedItem?.ToString() ?? "";
                    // Linear R² values are stored in percent -> convert to 0..1 for the Compare tab
                    MLRunHistory.AddRegression("Linear Regression", tgt, _result.TestR2 / 100.0, double.NaN, _result.TestMAE, _result.TestRMSE);
                }
                catch { }
            }
            catch (Exception ex)
            {
                if (resetVersion == _pageResetVersions[2]) TxtResults.Text = "Training failed:\n" + ex.Message;
            }
        }

        private void DisplayResults()
        {
            LinCvTable.Show(_linCv?.Report);
            if (_result == null) return;
            if (_linCv != null)
            {
                var cvsb = new StringBuilder();
                cvsb.AppendLine($"══════ {_linCv.K}-FOLD CV ══════");
                cvsb.AppendLine($"R²:   {_linCv.MeanR2:F2}% ± {_linCv.StdR2:F2}");
                cvsb.AppendLine($"RMSE: {_linCv.MeanRMSE:G4} ± {_linCv.StdRMSE:G4}");
                cvsb.AppendLine($"MAE:  {_linCv.MeanMAE:G4} ± {_linCv.StdMAE:G4}");
                cvsb.AppendLine();
                TxtResults.Text = cvsb.ToString();   // CV summary shown first; full-data metrics appended below
            }

            // Summary
            var sb = new StringBuilder();
            if (_linCv != null) sb.Append(TxtResults.Text);
            sb.AppendLine($"Status: {_result.ConvergenceMessage}");
            sb.AppendLine();
            sb.AppendLine(_linCv == null ? "══════ TRAIN SET ══════" : "══════ ALL-DATA FIT ══════");
            sb.AppendLine($"R²:     {_result.TrainR2:F4}%");
            sb.AppendLine($"MAE:    {_result.TrainMAE:G6}");
            sb.AppendLine($"RMSE:   {_result.TrainRMSE:G6}");
            sb.AppendLine();
            if (_linCv == null)
            {
                sb.AppendLine("══════ TEST SET ═══════");
                sb.AppendLine($"R²:     {_result.TestR2:F4}%");
                sb.AppendLine($"MAE:    {_result.TestMAE:G6}");
                sb.AppendLine($"RMSE:   {_result.TestRMSE:G6}");
            }
            sb.AppendLine();
            sb.AppendLine($"Bias:   {_result.Bias:G6}");
            TxtResults.Text = sb.ToString();

            // Weights
            var wsb = new StringBuilder();
            for (int i = 0; i < _result.Weights.Length; i++)
            {
                string name = i < _result.FeatureNames.Length ? _result.FeatureNames[i] : $"w[{i}]";
                wsb.AppendLine($"{name}: {_result.Weights[i]:G6}");
            }
            TxtWeights.Text = wsb.ToString();

            // Charts
            DrawCostChart();
            DrawScatterCharts();
            DrawResidualChart();
            DrawImportanceChart();
        }

        private void DrawCostChart()
        {
            var values = _result.CostHistory.Select((c, i) => new ObservablePoint(i, c)).ToList();

            CostChart.Series = new ISeries[]
            {
                new LineSeries<ObservablePoint>
                {
                    Values = values,
                    Fill = null,
                    Stroke = new SolidColorPaint(SKColor.Parse("#087F73"), 2),
                    GeometrySize = 0,
                    LineSmoothness = 0
                }
            };
            CostChart.XAxes = new Axis[]
            {
                new Axis
                {
                    Name = "Iteration",
                    NamePaint = new SolidColorPaint(SKColors.Gray),
                    LabelsPaint = new SolidColorPaint(SKColors.Gray),
                    TextSize = 11
                }
            };
            CostChart.YAxes = new Axis[]
            {
                new Axis
                {
                    Name = "Cost",
                    NamePaint = new SolidColorPaint(SKColors.Gray),
                    LabelsPaint = new SolidColorPaint(SKColors.Gray),
                    TextSize = 11
                }
            };
        }

        private void DrawScatterCharts()
        {
            // Train scatter
            var trainPts = new List<ObservablePoint>();
            for (int i = 0; i < _result.TrainActual.Length; i++)
                trainPts.Add(new ObservablePoint(_result.TrainActual[i], _result.TrainPredictions[i]));

            // Perfect line
            double minT = _result.TrainActual.Min(), maxT = _result.TrainActual.Max();
            var perfectTrain = new List<ObservablePoint>
            {
                new ObservablePoint(minT, minT),
                new ObservablePoint(maxT, maxT)
            };

            TrainScatter.Series = new ISeries[]
            {
                new ScatterSeries<ObservablePoint>
                {
                    Values = trainPts,
                    Fill = new SolidColorPaint(SKColor.Parse("#087F73")),
                    GeometrySize = 4,
                    Stroke = null
                },
                new LineSeries<ObservablePoint>
                {
                    Values = perfectTrain,
                    Fill = null,
                    Stroke = new SolidColorPaint(SKColors.OrangeRed, 2),
                    GeometrySize = 0,
                    LineSmoothness = 0
                }
            };
            SetScatterAxes(TrainScatter, "Actual", "Predicted");

            // Test scatter
            var testPts = new List<ObservablePoint>();
            for (int i = 0; i < _result.TestActual.Length; i++)
                testPts.Add(new ObservablePoint(_result.TestActual[i], _result.TestPredictions[i]));

            double minTe = _result.TestActual.Min(), maxTe = _result.TestActual.Max();
            var perfectTest = new List<ObservablePoint>
            {
                new ObservablePoint(minTe, minTe),
                new ObservablePoint(maxTe, maxTe)
            };

            TestScatter.Series = new ISeries[]
            {
                new ScatterSeries<ObservablePoint>
                {
                    Values = testPts,
                    Fill = new SolidColorPaint(SKColor.Parse("#ff6b6b")),
                    GeometrySize = 5,
                    Stroke = null
                },
                new LineSeries<ObservablePoint>
                {
                    Values = perfectTest,
                    Fill = null,
                    Stroke = new SolidColorPaint(SKColors.OrangeRed, 2),
                    GeometrySize = 0,
                    LineSmoothness = 0
                }
            };
            SetScatterAxes(TestScatter, "Actual", "Predicted");
        }

        private void DrawResidualChart()
        {
            var pts = new List<ObservablePoint>();
            for (int i = 0; i < _result.Residuals.Length; i++)
                pts.Add(new ObservablePoint(_result.TestPredictions[i], _result.Residuals[i]));

            // Zero line
            double minP = _result.TestPredictions.Min(), maxP = _result.TestPredictions.Max();
            var zeroLine = new List<ObservablePoint>
            {
                new ObservablePoint(minP, 0),
                new ObservablePoint(maxP, 0)
            };

            ResidualChart.Series = new ISeries[]
            {
                new ScatterSeries<ObservablePoint>
                {
                    Values = pts,
                    Fill = new SolidColorPaint(SKColor.Parse("#5275C5")),
                    GeometrySize = 4,
                    Stroke = null
                },
                new LineSeries<ObservablePoint>
                {
                    Values = zeroLine,
                    Fill = null,
                    Stroke = new SolidColorPaint(SKColors.Gray, 1),
                    GeometrySize = 0,
                    LineSmoothness = 0
                }
            };
            SetScatterAxes(ResidualChart, "Predicted", "Residual");
        }

        private void DrawImportanceChart()
        {
            var names = new List<string>();
            var vals = new List<double>();
            for (int i = 0; i < _result.Weights.Length; i++)
            {
                names.Add(i < _result.FeatureNames.Length ? _result.FeatureNames[i] : $"w{i}");
                vals.Add(Math.Abs(_result.Weights[i]));
            }

            // Sort by importance
            var sorted = names.Zip(vals, (n, v) => (n, v)).OrderByDescending(x => x.v).ToList();
            names = sorted.Select(x => x.n).ToList();
            vals = sorted.Select(x => x.v).ToList();

            ImportanceChart.Series = new ISeries[]
            {
                new RowSeries<double>
                {
                    Values = vals,
                    Fill = new SolidColorPaint(SKColor.Parse("#087F73")),
                    Stroke = null,
                    MaxBarWidth = 25
                }
            };
            ImportanceChart.YAxes = new Axis[]
            {
                new Axis
                {
                    Labels = names,
                    LabelsPaint = new SolidColorPaint(SKColors.Gray),
                    TextSize = 11
                }
            };
            ImportanceChart.XAxes = new Axis[]
            {
                new Axis
                {
                    Name = "|Weight|",
                    NamePaint = new SolidColorPaint(SKColors.Gray),
                    LabelsPaint = new SolidColorPaint(SKColors.Gray),
                    TextSize = 10
                }
            };
        }

        private void SetScatterAxes(LiveChartsCore.SkiaSharpView.WPF.CartesianChart chart,
            string xName, string yName)
        {
            chart.XAxes = new Axis[]
            {
                new Axis
                {
                    Name = xName,
                    NamePaint = new SolidColorPaint(SKColors.Gray),
                    LabelsPaint = new SolidColorPaint(SKColors.Gray),
                    TextSize = 10
                }
            };
            chart.YAxes = new Axis[]
            {
                new Axis
                {
                    Name = yName,
                    NamePaint = new SolidColorPaint(SKColors.Gray),
                    LabelsPaint = new SolidColorPaint(SKColors.Gray),
                    TextSize = 10
                }
            };
        }



        // ===================== PREPROCESSING TAB =====================

        private string _loadedCsvPath; // store path for missing value handling

        // Store path when loading CSV — ADD THIS LINE in LoadCsv_Click after successful load:
        // _loadedCsvPath = dlg.FileName;

        private void PrepStatus(string s) => TxtPrepStatus.Text = s;

        // ================ DESCRIBE ================
        private void Describe_Click(object sender, RoutedEventArgs e)
        {
            if (_dataset == null) { PrepStatus("Load data first!"); return; }

            try
            {
                var desc = PreprocessingEngine.Describe(_dataset);

                // Build DataTable for describe grid
                var dt = new DataTable();
                dt.Columns.Add("Statistic");
                foreach (var h in desc.Headers) dt.Columns.Add(h);

                string[] rowNames = { "Count", "Missing", "Mean", "Std Dev", "Min", "Q1 (25%)",
                              "Median", "Q3 (75%)", "Max", "Skewness", "Kurtosis" };

                for (int r = 0; r < rowNames.Length; r++)
                {
                    var row = dt.NewRow();
                    row[0] = rowNames[r];
                    for (int c = 0; c < desc.Headers.Length; c++)
                    {
                        switch (r)
                        {
                            case 0: row[c + 1] = desc.Count[c].ToString(); break;
                            case 1: row[c + 1] = desc.MissingCount[c].ToString(); break;
                            case 2: row[c + 1] = desc.Mean[c].ToString("G6"); break;
                            case 3: row[c + 1] = desc.StdDev[c].ToString("G6"); break;
                            case 4: row[c + 1] = desc.Min[c].ToString("G6"); break;
                            case 5: row[c + 1] = desc.Q1[c].ToString("G6"); break;
                            case 6: row[c + 1] = desc.Median[c].ToString("G6"); break;
                            case 7: row[c + 1] = desc.Q3[c].ToString("G6"); break;
                            case 8: row[c + 1] = desc.Max[c].ToString("G6"); break;
                            case 9: row[c + 1] = desc.Skewness[c].ToString("G4"); break;
                            case 10: row[c + 1] = desc.Kurtosis[c].ToString("G4"); break;
                        }
                    }
                    dt.Rows.Add(row);
                }

                DescribeGrid.ItemsSource = dt.DefaultView;
                DescribeBorder.Visibility = Visibility.Visible;
                PrepStatus($"Describe complete: {desc.Headers.Length} columns analyzed.");
            }
            catch (Exception ex) { PrepStatus("Describe error: " + ex.Message); }
        }



        // ================ MISSING VALUE DETECTION ================
        // ================ MISSING VALUE DETECTION ================
        private void DetectMissing_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_loadedCsvPath) || !File.Exists(_loadedCsvPath))
            {
                PrepStatus("⚠ Load a CSV file first from the Data tab!");
                return;
            }

            try
            {
                var lines = File.ReadAllLines(_loadedCsvPath)
                    .Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();

                if (lines.Length < 2)
                {
                    PrepStatus("CSV is empty or has only headers.");
                    return;
                }

                var headers = lines[0].Split(',').Select(h => h.Trim().Trim('"')).ToArray();
                int colCount = headers.Length;
                var missingPerCol = new int[colCount];
                var missingRows = new List<int>();
                int totalMissing = 0;

                for (int i = 1; i < lines.Length; i++)
                {
                    var values = lines[i].Split(',');
                    bool rowHasMissing = false;

                    for (int j = 0; j < colCount; j++)
                    {
                        string val = j < values.Length ? values[j].Trim() : "";
                        bool isMissing = string.IsNullOrWhiteSpace(val) ||
                                         val.Equals("NaN", StringComparison.OrdinalIgnoreCase) ||
                                         val.Equals("NA", StringComparison.OrdinalIgnoreCase) ||
                                         val.Equals("null", StringComparison.OrdinalIgnoreCase) ||
                                         val == "-" || val == "?";

                        // Also check if it's not a valid number
                        if (!isMissing && !double.TryParse(val, NumberStyles.Any,
                            CultureInfo.InvariantCulture, out _))
                        {
                            isMissing = true;
                        }

                        if (isMissing)
                        {
                            missingPerCol[j]++;
                            totalMissing++;
                            rowHasMissing = true;
                        }
                    }
                    if (rowHasMissing) missingRows.Add(i);
                }

                int dataRows = lines.Length - 1;
                bool hasMissing = totalMissing > 0;
                double missingPercent = dataRows > 0 ? (double)totalMissing / (dataRows * colCount) * 100 : 0;

                // Build report
                var sb = new StringBuilder();
                sb.AppendLine($"CSV Path:        {_loadedCsvPath}");
                sb.AppendLine($"Total Rows:      {dataRows}");
                sb.AppendLine($"Total Columns:   {colCount}");
                sb.AppendLine($"Total Missing:   {totalMissing}");
                sb.AppendLine($"Missing %:       {missingPercent:F2}%");
                sb.AppendLine($"Has Missing:     {(hasMissing ? "YES ⚠️" : "NO ✅")}");
                sb.AppendLine();

                if (hasMissing)
                {
                    sb.AppendLine("Missing Per Column:");
                    sb.AppendLine("─────────────────────────────");
                    for (int i = 0; i < headers.Length; i++)
                    {
                        int count = missingPerCol[i];
                        double pct = (double)count / dataRows * 100;
                        string bar = new string('█', Math.Min(count, 30));
                        sb.AppendLine($"  {headers[i],-18} {count,4} ({pct:F1}%)  {bar}");
                    }

                    sb.AppendLine();
                    if (missingRows.Count <= 20)
                        sb.AppendLine($"Rows with missing (1-based): {string.Join(", ", missingRows)}");
                    else
                        sb.AppendLine($"Rows with missing: {missingRows.Count} rows (first 10: {string.Join(", ", missingRows.Take(10))}...)");
                }

                TxtMissingReport.Text = sb.ToString();
                MissingBorder.Visibility = Visibility.Visible;

                // Chart
                if (hasMissing)
                {
                    var vals = missingPerCol.Select(x => (double)x).ToList();
                    MissingChart.Series = new ISeries[]
                    {
                new ColumnSeries<double>
                {
                    Values = vals,
                    Fill = new SolidColorPaint(SKColor.Parse("#ff6b6b")),
                    Stroke = null,
                    MaxBarWidth = 40
                }
                    };
                    MissingChart.XAxes = new Axis[]
                    {
                new Axis
                {
                    Labels = headers.ToList(),
                    LabelsPaint = new SolidColorPaint(SKColors.Gray),
                    TextSize = 10, LabelsRotation = 30
                }
                    };
                    MissingChart.YAxes = new Axis[]
                    {
                new Axis
                {
                    Name = "Missing Count",
                    NamePaint = new SolidColorPaint(SKColors.Gray),
                    LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10
                }
                    };
                    MissingChartBorder.Visibility = Visibility.Visible;
                }
                else
                {
                    MissingChartBorder.Visibility = Visibility.Collapsed;
                }

                PrepStatus(hasMissing
                    ? $"⚠ Found {totalMissing} missing values in {missingRows.Count} rows."
                    : "✅ No missing values found.");
            }
            catch (Exception ex) { PrepStatus("Detection error: " + ex.Message); }
        }

        // ================ MISSING VALUE HANDLING ================
        private void HandleMissing_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_loadedCsvPath) || !File.Exists(_loadedCsvPath))
            {
                PrepStatus("⚠ Load a CSV file first from the Data tab!");
                return;
            }

            try
            {
                var method = (MissingValueMethod)CmbMissingMethod.SelectedIndex;

                var lines = File.ReadAllLines(_loadedCsvPath)
                    .Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();

                if (lines.Length < 2) { PrepStatus("No data to clean."); return; }

                var headers = lines[0].Split(',').Select(h => h.Trim().Trim('"')).ToArray();
                int colCount = headers.Length;

                // Parse all data rows
                var data = new List<string[]>();
                for (int i = 1; i < lines.Length; i++)
                {
                    var vals = lines[i].Split(',').Select(v => v.Trim()).ToArray();
                    // Pad if row is shorter
                    if (vals.Length < colCount)
                    {
                        var padded = new string[colCount];
                        Array.Copy(vals, padded, vals.Length);
                        for (int j = vals.Length; j < colCount; j++) padded[j] = "";
                        data.Add(padded);
                    }
                    else
                    {
                        data.Add(vals.Take(colCount).ToArray());
                    }
                }

                // Compute column stats from valid numeric values
                var colValues = new List<double>[colCount];
                for (int j = 0; j < colCount; j++)
                {
                    colValues[j] = new List<double>();
                    foreach (var row in data)
                    {
                        if (double.TryParse(row[j], NumberStyles.Any,
                            CultureInfo.InvariantCulture, out double v))
                            colValues[j].Add(v);
                    }
                }

                var stats = new double[colCount];
                for (int j = 0; j < colCount; j++)
                {
                    if (colValues[j].Count == 0) { stats[j] = 0; continue; }
                    switch (method)
                    {
                        case MissingValueMethod.Mean: stats[j] = colValues[j].Average(); break;
                        case MissingValueMethod.Max: stats[j] = colValues[j].Max(); break;
                        case MissingValueMethod.Min: stats[j] = colValues[j].Min(); break;
                        case MissingValueMethod.Mode:
                            stats[j] = colValues[j].GroupBy(x => x)
                                .OrderByDescending(g => g.Count()).First().Key;
                            break;
                    }
                }

                // Check if value is missing
                bool IsMissing(string val)
                {
                    if (string.IsNullOrWhiteSpace(val)) return true;
                    if (val.Equals("NaN", StringComparison.OrdinalIgnoreCase)) return true;
                    if (val.Equals("NA", StringComparison.OrdinalIgnoreCase)) return true;
                    if (val.Equals("null", StringComparison.OrdinalIgnoreCase)) return true;
                    if (val == "-" || val == "?") return true;
                    if (!double.TryParse(val, NumberStyles.Any, CultureInfo.InvariantCulture, out _))
                        return true;
                    return false;
                }

                // Apply handling
                var cleaned = new List<string[]>();
                int totalFixed = 0;
                int removedRows = 0;

                foreach (var row in data)
                {
                    bool hasMissing = false;
                    var newRow = new string[colCount];

                    for (int j = 0; j < colCount; j++)
                    {
                        if (IsMissing(row[j]))
                        {
                            hasMissing = true;
                            if (method == MissingValueMethod.RemoveRow) break;
                            newRow[j] = stats[j].ToString("G10", CultureInfo.InvariantCulture);
                            totalFixed++;
                        }
                        else
                        {
                            newRow[j] = row[j];
                        }
                    }

                    if (method == MissingValueMethod.RemoveRow && hasMissing)
                    {
                        removedRows++;
                        continue;
                    }
                    cleaned.Add(newRow);
                }

                if (cleaned.Count == 0) { PrepStatus("No data remains after cleaning."); return; }

                // Save cleaned file
                string dir = Path.GetDirectoryName(_loadedCsvPath);
                string name = Path.GetFileNameWithoutExtension(_loadedCsvPath);
                string cleanPath = Path.Combine(dir, name + "_cleaned.csv");

                // Also save backup
                string backupPath = Path.Combine(dir, name + "_backup.csv");
                if (!File.Exists(backupPath))
                    File.Copy(_loadedCsvPath, backupPath, false);

                using (var w = new StreamWriter(cleanPath))
                {
                    w.WriteLine(string.Join(",", headers));
                    foreach (var row in cleaned)
                        w.WriteLine(string.Join(",", row));
                }

                // Reload dataset
                _dataset = MLDataset.LoadCsv(cleanPath);
                _loadedCsvPath = cleanPath;

                // Refresh data tab
                RefreshDataPreview();

                string msg;
                if (method == MissingValueMethod.RemoveRow)
                    msg = $"✅ Removed {removedRows} rows. {cleaned.Count} rows remain.\nSaved: {cleanPath}\nBackup: {backupPath}";
                else
                    msg = $"✅ Fixed {totalFixed} values using {method}.\nSaved: {cleanPath}\nBackup: {backupPath}";

                PrepStatus(msg);

                // Update missing report
                TxtMissingReport.Text = msg + "\n\n✅ Dataset is now clean!";
                MissingBorder.Visibility = Visibility.Visible;
                MissingChartBorder.Visibility = Visibility.Collapsed;
            }
            catch (Exception ex) { PrepStatus("Handling error: " + ex.Message); }
        }


        private void ResetAllTabs()
        {
            ResetWorkspacePage(2); ResetWorkspacePage(3); ResetWorkspacePage(4);
            for (int i = 0; i < _pageResetVersions.Length; i++) _pageResetVersions[i]++;
            _preprocessingSteps.Clear();
            ClearCrossValidationTables();
            // Reset fields
            _dataset = null;
            _loadedCsvPath = null;
            _result = null;

            ResetPolynomialResults();
            ResetVisualization();
            _normRecipe = null;
            _linModel = null;
            _linCv = null;
            _linInputs.Clear();
            LinPredictInputs.Children.Clear();
            LinPredictResult.Text = "";
            TrainScatter.Series = Array.Empty<ISeries>();
            TestScatter.Series = Array.Empty<ISeries>();
            TxtResults.Text = "Train a model to see results.";
            TxtWeights.Text = "";

            // ===== DATA TAB =====
            DataPreview.ItemsSource = null;
            TxtDataInfo.Text = "";
            TxtDataStatus.Text = "";
            TxtFileName.Text = "No file loaded";
            CmbStatCol.Items.Clear();
            CmbTarget.Items.Clear();
            CorrCanvas.Children.Clear();

            // Stats panel
            if (FindName("TxtStatCount") is TextBlock tc) tc.Text = "-";
            if (FindName("TxtStatMean") is TextBlock tm) tm.Text = "-";
            if (FindName("TxtStatStd") is TextBlock ts) ts.Text = "-";
            if (FindName("TxtStatMin") is TextBlock tmin) tmin.Text = "-";
            if (FindName("TxtStatMax") is TextBlock tmax) tmax.Text = "-";
            if (FindName("TxtStatMedian") is TextBlock tmed) tmed.Text = "-";
            if (FindName("TxtStatSkew") is TextBlock tsk) tsk.Text = "-";
            if (FindName("TxtStatKurt") is TextBlock tk) tk.Text = "-";
            if (FindName("HistogramChart") is CartesianChart hc) hc.Series = Array.Empty<ISeries>();

            // ===== PREPROCESSING TAB =====
            DescribeBorder.Visibility = Visibility.Collapsed;
            DescribeGrid.ItemsSource = null;

            MissingBorder.Visibility = Visibility.Collapsed;
            TxtMissingReport.Text = "";
            MissingChartBorder.Visibility = Visibility.Collapsed;
            MissingChart.Series = Array.Empty<ISeries>();

            AnomalyBorder.Visibility = Visibility.Collapsed;
            TxtAnomalyReport.Text = "";
            AnomalyChartBorder.Visibility = Visibility.Collapsed;
            AnomalyChart.Series = Array.Empty<ISeries>();
            TxtAnomalyThreshold.Text = "0.001";
            ChkRemoveAnomalies.IsChecked = false;

            NormalizeBorder.Visibility = Visibility.Collapsed;
            NormalizeGrid.ItemsSource = null;
            NormChartBorder.Visibility = Visibility.Collapsed;
            NormChart.Series = Array.Empty<ISeries>();
            CmbNormMethod.SelectedIndex = 0;

            PcaBorder.Visibility = Visibility.Collapsed;
            TxtPcaReport.Text = "";
            PcaScreeBorder.Visibility = Visibility.Collapsed;
            ScreeChart.Series = Array.Empty<ISeries>();
            PcaCumBorder.Visibility = Visibility.Collapsed;
            CumVarChart.Series = Array.Empty<ISeries>();
            PcaScatterBorder.Visibility = Visibility.Collapsed;
            PcaScatterChart.Series = Array.Empty<ISeries>();
            CmbPcaMode.SelectedIndex = 0;
            TxtPcaValue.Text = "0.95";
            ChkPcaHasTarget.IsChecked = true;


            EncoderBorder.Visibility = Visibility.Collapsed;
            TxtEncoderReport.Text = "";
            EncoderPreviewBorder.Visibility = Visibility.Collapsed;
            EncoderGrid.ItemsSource = null;
            TxtEncoderCols.Text = "";

            TxtPrepStatus.Text = "Load data in Data tab first.";

            // ===== LINEAR REGRESSION TAB =====
            if (FindName("CostChart") is CartesianChart cc) cc.Series = Array.Empty<ISeries>();
            if (FindName("ScatterChart") is CartesianChart sc) sc.Series = Array.Empty<ISeries>();
            if (FindName("ResidualChart") is CartesianChart rc) rc.Series = Array.Empty<ISeries>();
            if (FindName("ImportanceChart") is CartesianChart ic) ic.Series = Array.Empty<ISeries>();

            if (FindName("TxtMSE") is TextBlock mse) mse.Text = "-";
            if (FindName("TxtR2") is TextBlock r2) r2.Text = "-";
            if (FindName("TxtMAE") is TextBlock mae) mae.Text = "-";
            if (FindName("TxtRMSE") is TextBlock rmse) rmse.Text = "-";
            if (FindName("TxtEquation") is TextBlock eq) eq.Text = "";
            if (FindName("TxtTrainStatus") is TextBlock trs) trs.Text = "";
        }
        // ================ ANOMALY DETECTION ================
        private void Anomaly_Click(object sender, RoutedEventArgs e)
        {
            if (_dataset == null) { PrepStatus("Load data first!"); return; }

            if (!double.TryParse(TxtAnomalyThreshold.Text, NumberStyles.Any,
                CultureInfo.InvariantCulture, out double threshold) || threshold <= 0)
            { PrepStatus("Invalid threshold."); return; }

            try
            {
                var result = PreprocessingEngine.DetectAnomalies(_dataset, threshold);

                var sb = new StringBuilder();
                sb.AppendLine($"Total Samples:    {result.TotalRows}");
                sb.AppendLine($"Anomalies Found:  {result.AnomalyCount}");
                sb.AppendLine($"Threshold:        {result.Threshold:E4}");
                sb.AppendLine($"Anomaly Rate:     {(double)result.AnomalyCount / result.TotalRows * 100:F2}%");
                sb.AppendLine();

                if (result.AnomalyCount > 0)
                {
                    sb.AppendLine("Anomaly Indices:");
                    if (result.AnomalyIndices.Length <= 30)
                        sb.AppendLine("  " + string.Join(", ", result.AnomalyIndices));
                    else
                        sb.AppendLine($"  {result.AnomalyIndices.Length} anomalies (first 10: {string.Join(", ", result.AnomalyIndices.Take(10))}...)");

                    sb.AppendLine();
                    sb.AppendLine("Top 5 highest Mahalanobis distances:");
                    var top = result.MahalanobisDistances
                        .Select((d, i) => (d, i))
                        .OrderByDescending(x => x.d)
                        .Take(5);
                    foreach (var (d, i) in top)
                        sb.AppendLine($"  Row {i}: {d:F4}");
                }

                TxtAnomalyReport.Text = sb.ToString();
                AnomalyBorder.Visibility = Visibility.Visible;

                // Mahalanobis distance chart
                var allDists = result.MahalanobisDistances.ToList();
                var normalPts = new List<ObservablePoint>();
                var anomalyPts = new List<ObservablePoint>();

                for (int i = 0; i < allDists.Count; i++)
                {
                    if (result.AnomalyIndices.Contains(i))
                        anomalyPts.Add(new ObservablePoint(i, allDists[i]));
                    else
                        normalPts.Add(new ObservablePoint(i, allDists[i]));
                }

                AnomalyChart.Series = new ISeries[]
                {
            new ScatterSeries<ObservablePoint>
            {
                Values = normalPts, GeometrySize = 4,
                Fill = new SolidColorPaint(SKColor.Parse("#087F73")),
                Stroke = null
            },
            new ScatterSeries<ObservablePoint>
            {
                Values = anomalyPts, GeometrySize = 7,
                Fill = new SolidColorPaint(SKColor.Parse("#ff6b6b")),
                Stroke = null
            }
                };
                AnomalyChart.XAxes = new Axis[]
                {
            new Axis { Name = "Sample Index",
                NamePaint = new SolidColorPaint(SKColors.Gray),
                LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 }
                };
                AnomalyChart.YAxes = new Axis[]
                {
            new Axis { Name = "Mahalanobis Distance",
                NamePaint = new SolidColorPaint(SKColors.Gray),
                LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 }
                };
                AnomalyChartBorder.Visibility = Visibility.Visible;

                // If remove anomalies checked, update dataset
                if (ChkRemoveAnomalies.IsChecked == true && result.FilteredData.Length > 0)
                {
                    // Rebuild dataset from filtered data
                    string dir = Path.GetDirectoryName(_loadedCsvPath);
                    string name = Path.GetFileNameWithoutExtension(_loadedCsvPath);
                    string filtPath = Path.Combine(dir, name + "_filtered.csv");

                    using (var w = new StreamWriter(filtPath))
                    {
                        w.WriteLine(string.Join(",", _dataset.Headers));
                        foreach (var row in result.FilteredData)
                            w.WriteLine(string.Join(",", row.Select(v =>
                                v.ToString("G10", CultureInfo.InvariantCulture))));
                    }

                    _dataset = MLDataset.LoadCsv(filtPath);
                    _loadedCsvPath = filtPath;
                    RefreshDataPreview();
                    PrepStatus($"Anomalies removed. {result.FilteredData.Length} rows remain. Saved: {filtPath}");
                }
                else
                {
                    PrepStatus($"Found {result.AnomalyCount} anomalies out of {result.TotalRows} samples.");
                }
            }
            catch (Exception ex) { PrepStatus("Anomaly error: " + ex.Message); }
        }

        // ================ NORMALIZE ================
        private void Normalize_Click(object sender, RoutedEventArgs e)
        {
            if (_dataset == null) { PrepStatus("Load data first!"); return; }

            try
            {
                var method = (NormMethod)CmbNormMethod.SelectedIndex;
                var result = PreprocessingEngine.NormalizeData(_dataset, method);
                var normalization = NormalizationRecipe.ForDataset(_dataset, method);

                // Show preview table
                var dt = new DataTable();
                foreach (var h in result.Headers) dt.Columns.Add(h);
                int preview = Math.Min(result.NormalizedData.Length, 200);
                for (int i = 0; i < preview; i++)
                {
                    var row = dt.NewRow();
                    for (int j = 0; j < result.Headers.Length; j++)
                        row[j] = result.NormalizedData[i][j].ToString("F6");
                    dt.Rows.Add(row);
                }
                NormalizeGrid.ItemsSource = dt.DefaultView;
                TxtNormTitle.Text = $"Normalized Data — {result.MethodName}";
                NormalizeBorder.Visibility = Visibility.Visible;

                // Before/After range chart
                var beforeMin = new List<double>();
                var beforeMax = new List<double>();
                var afterMin = new List<double>();
                var afterMax = new List<double>();
                var labels = new List<string>();

                for (int c = 0; c < result.Headers.Length; c++)
                {
                    labels.Add(result.Headers[c]);
                    beforeMin.Add(_dataset.Min(c));
                    beforeMax.Add(_dataset.Max(c));
                    var col = result.NormalizedData.Select(r => r[c]).ToArray();
                    afterMin.Add(col.Min());
                    afterMax.Add(col.Max());
                }

                var beforeRange = beforeMin.Zip(beforeMax, (mn, mx) => mx - mn).ToList();
                var afterRange = afterMin.Zip(afterMax, (mn, mx) => mx - mn).ToList();

                NormChart.Series = new ISeries[]
                {
            new ColumnSeries<double>
            {
                Values = beforeRange, Name = "Before",
                Fill = new SolidColorPaint(SKColor.Parse("#A8A29B")),
                MaxBarWidth = 20
            },
            new ColumnSeries<double>
            {
                Values = afterRange, Name = "After",
                Fill = new SolidColorPaint(SKColor.Parse("#087F73")),
                MaxBarWidth = 20
            }
                };
                NormChart.XAxes = new Axis[]
                {
            new Axis { Labels = labels,
                LabelsPaint = new SolidColorPaint(SKColors.Gray),
                TextSize = 10, LabelsRotation = 30 }
                };
                NormChart.YAxes = new Axis[]
                {
            new Axis { Name = "Range",
                NamePaint = new SolidColorPaint(SKColors.Gray),
                LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 }
                };
                NormChartBorder.Visibility = Visibility.Visible;

                // Apply normalized data to the dataset so all algorithms use it
                string ndir = Path.GetDirectoryName(_loadedCsvPath);
                string nname = Path.GetFileNameWithoutExtension(_loadedCsvPath);
                string normPath = Path.Combine(ndir, nname + "_normalized.csv");
                using (var w = new StreamWriter(normPath))
                {
                    w.WriteLine(string.Join(",", result.Headers));
                    foreach (var row in result.NormalizedData)
                        w.WriteLine(string.Join(",", row.Select(v =>
                            v.ToString("G10", CultureInfo.InvariantCulture))));
                }
                _dataset = MLDataset.LoadCsv(normPath);
                _preprocessingSteps.Add(new PreprocessingStep { Normalization = normalization });
                _loadedCsvPath = normPath;
                RefreshDataPreview();
                PrepStatus($"Normalization applied to dataset ({result.MethodName}). Saved: {normPath}");
            }
            catch (Exception ex) { PrepStatus("Normalize error: " + ex.Message); }
        }

        // ================ PCA ================
        private async void Pca_Click(object sender, RoutedEventArgs e)
        {
            if (_dataset == null) { PrepStatus("Load data first!"); return; }

            if (!double.TryParse(TxtPcaValue.Text, NumberStyles.Any,
                CultureInfo.InvariantCulture, out double value))
            { PrepStatus("Invalid PCA value."); return; }

            int mode = CmbPcaMode.SelectedIndex;
            bool hasTarget = ChkPcaHasTarget.IsChecked == true;
            int resetVersion = _pageResetVersions[1];
            var sourceData = _dataset;

            PrepStatus("Running PCA...");

            try
            {
                var result = await Task.Run(() =>
                    PreprocessingEngine.RunPCA(sourceData, mode, value, hasTarget));
                if (resetVersion != _pageResetVersions[1] || !ReferenceEquals(sourceData, _dataset)) return;

                // Report
                var sb = new StringBuilder();
                sb.AppendLine($"Original Features:     {result.OriginalHeaders.Length}");
                sb.AppendLine($"Components Retained:   {result.ComponentsKept}");
                sb.AppendLine($"Variance Retained:     {result.VarianceRetained:F2}%");
                sb.AppendLine();
                sb.AppendLine("Eigenvalues:");
                sb.AppendLine("─────────────────────────");
                for (int i = 0; i < result.Eigenvalues.Length; i++)
                {
                    string mark = i < result.ComponentsKept ? "✓" : " ";
                    sb.AppendLine($"  {mark} PC{i + 1}: {result.Eigenvalues[i]:G6}" +
                        $"  ({result.ExplainedVarianceRatio[i] * 100:F2}%)");
                }
                sb.AppendLine();
                sb.AppendLine("Feature Contributions (PC1):");
                sb.AppendLine("─────────────────────────");
                if (result.Eigenvectors.Length > 0)
                {
                    for (int i = 0; i < result.OriginalHeaders.Length; i++)
                        sb.AppendLine($"  {result.OriginalHeaders[i],-15} {result.Eigenvectors[i][0]:F4}");
                }

                TxtPcaReport.Text = sb.ToString();
                PcaBorder.Visibility = Visibility.Visible;

                // Scree plot
                var evrPercent = result.ExplainedVarianceRatio.Select(v => v * 100).ToList();
                var pcLabels = Enumerable.Range(1, evrPercent.Count).Select(i => $"PC{i}").ToList();

                ScreeChart.Series = new ISeries[]
                {
            new ColumnSeries<double>
            {
                Values = evrPercent,
                Fill = new SolidColorPaint(SKColor.Parse("#087F73")),
                Stroke = null, MaxBarWidth = 30
            }
                };
                ScreeChart.XAxes = new Axis[]
                {
            new Axis { Labels = pcLabels,
                LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 11 }
                };
                ScreeChart.YAxes = new Axis[]
                {
            new Axis { Name = "Variance %",
                NamePaint = new SolidColorPaint(SKColors.Gray),
                LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 }
                };
                PcaScreeBorder.Visibility = Visibility.Visible;

                // Cumulative variance
                var cumPts = result.CumulativeVariance
                    .Select((v, i) => new ObservablePoint(i + 1, v * 100)).ToList();

                // Threshold line
                double thresh = mode == 0 ? value * 100 : result.VarianceRetained;
                var threshLine = new List<ObservablePoint>
        {
            new ObservablePoint(1, thresh),
            new ObservablePoint(result.CumulativeVariance.Length, thresh)
        };

                CumVarChart.Series = new ISeries[]
                {
            new LineSeries<ObservablePoint>
            {
                Values = cumPts, Fill = null,
                Stroke = new SolidColorPaint(SKColor.Parse("#087F73"), 2),
                GeometrySize = 6,
                GeometryFill = new SolidColorPaint(SKColor.Parse("#087F73")),
                LineSmoothness = 0
            },
            new LineSeries<ObservablePoint>
            {
                Values = threshLine, Fill = null,
                Stroke = new SolidColorPaint(SKColor.Parse("#ff6b6b"), 1.5f)
                { PathEffect = new LiveChartsCore.SkiaSharpView.Painting.Effects.DashEffect(new float[] { 6, 4 }) },
                GeometrySize = 0, LineSmoothness = 0
            }
                };
                CumVarChart.XAxes = new Axis[]
                {
            new Axis { Name = "Component",
                NamePaint = new SolidColorPaint(SKColors.Gray),
                LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 }
                };
                CumVarChart.YAxes = new Axis[]
                {
            new Axis { Name = "Cumulative Variance %",
                NamePaint = new SolidColorPaint(SKColors.Gray),
                LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10,
                MinLimit = 0, MaxLimit = 105 }
                };
                PcaCumBorder.Visibility = Visibility.Visible;

                // 2D scatter if at least 2 components
                if (result.ComponentsKept >= 2)
                {
                    var pts = result.ProjectedData
                        .Select(r => new ObservablePoint(r[0], r[1])).ToList();

                    PcaScatterChart.Series = new ISeries[]
                    {
                new ScatterSeries<ObservablePoint>
                {
                    Values = pts, GeometrySize = 4,
                    Fill = new SolidColorPaint(SKColor.Parse("#5275C5")),
                    Stroke = null
                }
                    };
                    PcaScatterChart.XAxes = new Axis[]
                    {
                new Axis { Name = "PC1",
                    NamePaint = new SolidColorPaint(SKColors.Gray),
                    LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 }
                    };
                    PcaScatterChart.YAxes = new Axis[]
                    {
                new Axis { Name = "PC2",
                    NamePaint = new SolidColorPaint(SKColors.Gray),
                    LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 }
                    };
                    PcaScatterBorder.Visibility = Visibility.Visible;
                }

                // Apply PCA output to the dataset: PC1..PCk (+ target if last column is target)
                int k = result.ComponentsKept;
                int targetIdx = _dataset.ColCount - 1;
                var pcaHeaders = new List<string>();
                for (int c = 0; c < k; c++) pcaHeaders.Add("PC" + (c + 1));
                if (hasTarget) pcaHeaders.Add(_dataset.Headers[targetIdx]);

                var pcaLines = new List<string>();
                int rowN = Math.Min(result.ProjectedData.Length, _dataset.RowCount);
                for (int i = 0; i < rowN; i++)
                {
                    var vals = new List<string>();
                    for (int c = 0; c < k; c++)
                        vals.Add(result.ProjectedData[i][c].ToString("G10", CultureInfo.InvariantCulture));
                    if (hasTarget)
                        vals.Add(_dataset.Rows[i][targetIdx].ToString("G10", CultureInfo.InvariantCulture));
                    pcaLines.Add(string.Join(",", vals));
                }

                string pdir = Path.GetDirectoryName(_loadedCsvPath);
                string pname = Path.GetFileNameWithoutExtension(_loadedCsvPath);
                string pcaPath = Path.Combine(pdir, pname + "_pca.csv");
                using (var w = new StreamWriter(pcaPath))
                {
                    w.WriteLine(string.Join(",", pcaHeaders));
                    foreach (var line in pcaLines) w.WriteLine(line);
                }
                _dataset = MLDataset.LoadCsv(pcaPath);
                _preprocessingSteps.Add(PreprocessingStep.FromPca(result));
                _loadedCsvPath = pcaPath;
                RefreshDataPreview();
                PrepStatus($"PCA applied to dataset: {k} components ({result.VarianceRetained:F1}% variance). Saved: {pcaPath}");
            }
            catch (Exception ex) { if (resetVersion == _pageResetVersions[1]) PrepStatus("PCA error: " + ex.Message); }
        }

        // ================ HELPER: Refresh Data Preview ================
        private void RefreshDataPreview()
        {
            ClearCrossValidationTables();
            if (_dataset == null) return;

            ResetPolynomialResults();
            ResetVisualization();
            RefreshPolyCombos();
            RefreshVizCombos();

            var dt = new DataTable();
            foreach (var h in _dataset.Headers) dt.Columns.Add(h);
            int previewCount = Math.Min(_dataset.RowCount, 500);
            for (int i = 0; i < previewCount; i++)
            {
                var row = dt.NewRow();
                for (int j = 0; j < _dataset.ColCount; j++)
                    row[j] = _dataset.Rows[i][j].ToString("G6", CultureInfo.InvariantCulture);
                dt.Rows.Add(row);
            }
            DataPreview.ItemsSource = dt.DefaultView;

            // Update combos
            CmbStatCol.Items.Clear();
            CmbTarget.Items.Clear();
            foreach (var h in _dataset.Headers)
            {
                CmbStatCol.Items.Add(h);
                CmbTarget.Items.Add(h);
            }
            if (CmbStatCol.Items.Count > 0) CmbStatCol.SelectedIndex = 0;
            if (CmbTarget.Items.Count > 0) CmbTarget.SelectedIndex = _dataset.ColCount - 1;

            TxtDataInfo.Text = $"Rows: {_dataset.RowCount}\nColumns: {_dataset.ColCount}";
        }

        // ===================== HELPERS =====================

        private FormattedText MkTxt(string text, double size, Brush brush)
        {
            double dpi = 96;
            try { dpi = VisualTreeHelper.GetDpi(this).PixelsPerDip; } catch { }
            return new FormattedText(text, CultureInfo.InvariantCulture,
                FlowDirection.LeftToRight, new Typeface("Segoe UI"), size, brush, dpi);
        }

        private void MainTabs_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {

        }

        private void MainTabs_SelectionChanged_1(object sender, SelectionChangedEventArgs e)
        {

        }
    }

    // Reuse from ElevationView or add here
   
}
