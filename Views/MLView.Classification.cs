using ArchiAI.Helpers;
using ArchiAI.ML;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Data;
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
        private ClassificationResult _clsResult;
        private ClassificationCVResult _clsCv;
        private readonly List<TextBox> _clsPredictBoxes = new();

        // Populate the class-target combo when the Classification tab is first shown.
        private void ClsPanel_IsVisibleChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            if (!(bool)e.NewValue || _dataset == null || CmbClsTarget == null) return;
            if (CmbClsTarget.Items.Cast<string>().SequenceEqual(_dataset.Headers)) return;
            CmbClsTarget.Items.Clear();
            foreach (var h in _dataset.Headers) CmbClsTarget.Items.Add(h);
            CmbClsTarget.SelectedIndex = _dataset.ColCount - 1;
        }

        private void ClsAlgo_Changed(object sender, SelectionChangedEventArgs e)
        {
            if (ClsLogRegParams == null || ClsKnnParams == null) return;
            int i = CmbClsAlgo.SelectedIndex; // 0 LogReg,1 KNN,2 NB,3 SVM,4 RF,5 FastTree,6 LightGBM,7 XGBoost,8 ANN
            // Iterations + Learning Rate are used by LogReg, SVM, boosted (LR) and ANN (epochs)
            ClsLogRegParams.Visibility = (i == 0 || i == 3 || i == 5 || i == 6 || i == 7 || i == 8) ? Visibility.Visible : Visibility.Collapsed;
            ClsKnnParams.Visibility = i == 1 ? Visibility.Visible : Visibility.Collapsed;
            if (ClsSvmParams != null) ClsSvmParams.Visibility = i == 3 ? Visibility.Visible : Visibility.Collapsed;
            // trees/depth/leaf used by Random Forest + boosted (FastTree/LightGBM/XGBoost)
            if (ClsRfParams != null) ClsRfParams.Visibility = (i == 4 || i == 5 || i == 6 || i == 7) ? Visibility.Visible : Visibility.Collapsed;
            if (ClsAnnParams != null) ClsAnnParams.Visibility = i == 8 ? Visibility.Visible : Visibility.Collapsed;
        }

        private void ClsSplit_Changed(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (TxtClsSplit != null)
            {
                int train = (int)SliderClsSplit.Value;
                TxtClsSplit.Text = $"{train}/{100 - train}";
            }
        }

        private async void ClsTrain_Click(object sender, RoutedEventArgs e)
        {
            if (_dataset == null) { TxtClsResults.Text = "Load data first!"; return; }
            // ensure target combo is populated
            if (!CmbClsTarget.Items.Cast<string>().SequenceEqual(_dataset.Headers))
            {
                CmbClsTarget.Items.Clear();
                foreach (var h in _dataset.Headers) CmbClsTarget.Items.Add(h);
                CmbClsTarget.SelectedIndex = _dataset.ColCount - 1;
            }
            int targetCol = CmbClsTarget.SelectedIndex;
            if (targetCol < 0) { TxtClsResults.Text = "Select a target (class) column."; return; }

            var featureNames = new List<string>();
            for (int i = 0; i < _dataset.ColCount; i++)
                if (i != targetCol) featureNames.Add(_dataset.Headers[i]);

            var engine = new ClassificationEngine
            {
                Type = (ClassifierType)CmbClsAlgo.SelectedIndex,
                TrainTestSplit = SliderClsSplit.Value / 100.0
            };
            if (!int.TryParse(TxtClsIter.Text, out int iters) || iters < 1) iters = 500;
            if (!double.TryParse(TxtClsLR.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out double lr) || lr <= 0) lr = 0.1;
            if (!int.TryParse(TxtClsK.Text, out int kn) || kn < 1) kn = 5;
            engine.Iterations = iters; engine.LearningRate = lr; engine.Neighbors = kn;
            // SVM + Random Forest params (fall back to defaults if boxes empty/invalid)
            if (ClsSvmParams != null && double.TryParse(TxtClsC.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out double svmC) && svmC > 0) engine.SvmC = svmC;
            if (TxtClsTrees != null && int.TryParse(TxtClsTrees.Text, out int trees) && trees > 0) engine.Trees = trees;
            if (TxtClsDepth != null && int.TryParse(TxtClsDepth.Text, out int depth) && depth > 0) engine.MaxDepth = depth;
            if (TxtClsLeaf != null && int.TryParse(TxtClsLeaf.Text, out int leaf) && leaf > 0) engine.MinLeaf = leaf;
            if (TxtClsAnnHidden != null && int.TryParse(TxtClsAnnHidden.Text, out int ah) && ah > 0) engine.AnnHidden = ah;
            if (TxtClsAnnLayers != null && int.TryParse(TxtClsAnnLayers.Text, out int al) && al > 0) engine.AnnLayers = al;

            bool useCv = ChkClsKFold.IsChecked == true;
            int folds = ParseFolds(TxtClsFolds.Text);
            ClsCvTable.Show(null);
            TxtClsResults.Text = "Training...";
            int resetVersion = _pageResetVersions[3];
            var trainingData = _dataset;
            try
            {
                var pipeline = CapturePredictionRecipe(targetCol);
                if (useCv)
                {
                    var cv = await Task.Run(() => engine.CrossValidate(trainingData, targetCol, featureNames.ToArray(), folds));
                    if (resetVersion != _pageResetVersions[3]) return;
                    _clsCv = cv;
                    _clsResult = cv.FinalModel;
                }
                else
                {
                    _clsCv = null;
                    var result = await Task.Run(() => engine.Run(trainingData, targetCol, featureNames.ToArray()));
                    if (resetVersion != _pageResetVersions[3]) return;
                    _clsResult = result;
                }
                _clsResult.Pipeline = pipeline;
                _clsResult.TargetName = pipeline.TargetName;
                DisplayClsResults();
                BuildClsPredictInputs();
                try
                {
                    string algo = (CmbClsAlgo.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "Classifier";
                    string tgt = CmbClsTarget.SelectedItem?.ToString() ?? "";
                    MLRunHistory.AddClassification(algo, tgt, _clsResult.TestAccuracy, _clsResult.MacroF1);
                }
                catch { }
            }
            catch (Exception ex)
            {
                if (resetVersion == _pageResetVersions[3]) TxtClsResults.Text = "Training failed:\n" + ex.Message;
            }
        }

        private void DisplayClsResults()
        {
            ClsCvTable.Show(_clsCv?.Report);
            if (_clsResult == null) return;
            var r = _clsResult;
            var sb = new StringBuilder();
            if (_clsCv != null)
            {
                sb.AppendLine($"══════ {_clsCv.K}-FOLD CV ══════");
                sb.AppendLine($"Accuracy: {_clsCv.MeanAccuracy:F2}% ± {_clsCv.StdAccuracy:F2}");
                sb.AppendLine($"Macro-F1: {_clsCv.MeanF1:F3} ± {_clsCv.StdF1:F3}");
                sb.AppendLine();
            }
            sb.AppendLine($"Model: {r.ConvergenceMessage}");
            sb.AppendLine($"Classes: {string.Join(", ", r.Classes.Select(c => c.ToString("G6")))}");
            sb.AppendLine();
            if (_clsCv != null)
                sb.AppendLine($"Final all-data fit accuracy: {r.TrainAccuracy:F2}% (training only)");
            else
            {
                sb.AppendLine($"Train Accuracy: {r.TrainAccuracy:F2}%");
                sb.AppendLine($"Test Accuracy:  {r.TestAccuracy:F2}%");
            }
            sb.AppendLine($"Macro-F1:       {r.MacroF1:F3}");
            TxtClsResults.Text = sb.ToString();

            // per-class metrics grid
            var dt = new DataTable();
            dt.Columns.Add("Class");
            dt.Columns.Add("Precision");
            dt.Columns.Add("Recall");
            dt.Columns.Add("F1");
            for (int c = 0; c < r.Classes.Length; c++)
            {
                var row = dt.NewRow();
                row[0] = r.Classes[c].ToString("G6");
                row[1] = r.Precision[c].ToString("F3");
                row[2] = r.Recall[c].ToString("F3");
                row[3] = r.F1[c].ToString("F3");
                dt.Rows.Add(row);
            }
            ClsMetricsGrid.ItemsSource = dt.DefaultView;

            // per-class F1 bar chart
            var f1vals = r.F1.ToList();
            var labels = r.Classes.Select(c => c.ToString("G4")).ToList();
            ClsF1Chart.Series = new ISeries[]
            {
                new ColumnSeries<double>
                {
                    Values = f1vals,
                    Fill = new SolidColorPaint(SKColor.Parse("#087F73")),
                    Stroke = null, MaxBarWidth = 40
                }
            };
            ClsF1Chart.XAxes = new Axis[]
            {
                new Axis { Labels = labels, LabelsPaint = new SolidColorPaint(SKColors.Gray),
                    TextSize = 11, LabelsRotation = 15 }
            };
            ClsF1Chart.YAxes = new Axis[]
            {
                new Axis { Name = "F1", NamePaint = new SolidColorPaint(SKColors.Gray),
                    LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10, MinLimit = 0, MaxLimit = 1 }
            };

            DrawConfusionMatrix(r);
        }

        private void DrawConfusionMatrix(ClassificationResult r)
        {
            ClsConfCanvas.Children.Clear();
            int n = r.Classes.Length;
            int max = 0;
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (r.Confusion[i, j] > max) max = r.Confusion[i, j];
            if (max == 0) max = 1;

            double canvasW = ClsConfCanvas.ActualWidth > 10 ? ClsConfCanvas.ActualWidth : 520;
            double labelW = 90, top = 46;
            double cell = Math.Min((canvasW - labelW) / n, (300.0) / n);
            cell = Math.Max(28, cell);
            ClsConfCanvas.Height = top + n * cell + 20;

            var dv = new DrawingVisual();
            using (var dc = dv.RenderOpen())
            {
                dc.DrawText(MkTxt("Predicted →", 11, Brushes.Gray), new Point(labelW, 4));
                dc.DrawText(MkTxt("Actual ↓", 11, Brushes.Gray), new Point(4, 24));
                for (int j = 0; j < n; j++)
                {
                    var ft = MkTxt(r.Classes[j].ToString("G4"), 10, Brushes.LightGray);
                    dc.DrawText(ft, new Point(labelW + j * cell + cell / 2 - ft.Width / 2, top - 18));
                }
                for (int i = 0; i < n; i++)
                {
                    var rl = MkTxt(r.Classes[i].ToString("G4"), 10, Brushes.LightGray);
                    dc.DrawText(rl, new Point(labelW - rl.Width - 6, top + i * cell + cell / 2 - rl.Height / 2));
                    for (int j = 0; j < n; j++)
                    {
                        int v = r.Confusion[i, j];
                        double t = (double)v / max;
                        // teal intensity; diagonal (correct) brighter
                        byte g = (byte)(40 + t * 200);
                        Color col = (i == j) ? Color.FromRgb((byte)(20), g, (byte)(180 * t + 40))
                                             : Color.FromRgb((byte)(60 + 120 * t), (byte)(40 + 40 * t), (byte)(70 + 90 * t));
                        var rect = new Rect(labelW + j * cell, top + i * cell, cell - 2, cell - 2);
                        dc.DrawRectangle(new SolidColorBrush(col), null, rect);
                        var vt = MkTxt(v.ToString(), Math.Max(10, cell * 0.32),
                            new SolidColorBrush(t > 0.4 ? Colors.White : Colors.LightGray));
                        dc.DrawText(vt, new Point(rect.X + rect.Width / 2 - vt.Width / 2,
                                                  rect.Y + rect.Height / 2 - vt.Height / 2));
                    }
                }
            }
            ClsConfCanvas.Children.Add(new DrawingVisualHost(dv));
        }

        private void BuildClsPredictInputs()
        {
            ClsPredictInputs.Children.Clear();
            _clsPredictBoxes.Clear();
            if (_clsResult == null) return;
            foreach (var name in _clsResult.InputNames)
            {
                var lbl = new TextBlock { Text = name, Foreground = ClsB("#334155"), FontSize = 12, Margin = new Thickness(0, 6, 0, 2) };
                var tb = new TextBox { Text = "0", FontSize = 13, Background = ClsB("#F1F5F9"), Foreground = Brushes.DarkSlateGray, BorderBrush = ClsB("#E2E7EF"), Padding = new Thickness(6, 4, 6, 4) };
                _clsPredictBoxes.Add(tb);
                ClsPredictInputs.Children.Add(lbl);
                ClsPredictInputs.Children.Add(tb);
            }
        }

        private void ClsPredict_Click(object sender, RoutedEventArgs e)
        {
            if (_clsResult == null) { ClsPredictResult.Text = "Train a model first."; return; }
            var x = new double[_clsPredictBoxes.Count];
            for (int i = 0; i < x.Length; i++)
            {
                if (!double.TryParse(_clsPredictBoxes[i].Text.Trim(), NumberStyles.Any, CultureInfo.InvariantCulture, out x[i]))
                { ClsPredictResult.Text = $"Invalid value for {_clsResult.InputNames[i]}"; return; }
            }
            try
            {
                double cls = _clsResult.PredictRaw(x);
                ClsPredictResult.Text = $"{_clsResult.TargetName ?? "Predicted class"}: {cls:G6}";
            }
            catch (Exception ex) { ClsPredictResult.Text = "Predict error: " + ex.Message; }
        }

        private void ClsSaveModel_Click(object sender, RoutedEventArgs e)
        {
            if (_clsResult?.Model == null) { ClsPredictResult.Text = "Train a model first."; return; }
            var dlg = new Microsoft.Win32.SaveFileDialog { Filter = "ArchiAI Classifier|*.aiclf", FileName = "classifier.aiclf" };
            if (dlg.ShowDialog() == true)
            {
                try { ClassificationIO.Save(_clsResult, _clsResult.FeatureNames, CmbClsTarget.SelectedItem?.ToString() ?? "", dlg.FileName); ClsPredictResult.Text = "✅ Saved: " + dlg.FileName; }
                catch (Exception ex) { ClsPredictResult.Text = "Save error: " + ex.Message; }
            }
        }

        private void ClsLoadModel_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new Microsoft.Win32.OpenFileDialog { Filter = "ArchiAI Classifier|*.aiclf" };
            if (dlg.ShowDialog() != true) return;
            try
            {
                _clsResult = ClassificationIO.Load(dlg.FileName, out _, out _);
                ClsCvTable.Show(null);
                _clsCv = null;
                BuildClsPredictInputs();
                ClsPredictResult.Text = "✅ Model loaded — enter feature values and Predict.";
            }
            catch (Exception ex) { ClsPredictResult.Text = "Load error: " + ex.Message; }
        }

        private static SolidColorBrush ClsB(string hex) => new SolidColorBrush((Color)ColorConverter.ConvertFromString(hex));
    }
}
