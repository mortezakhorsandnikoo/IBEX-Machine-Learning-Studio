using ArchiAI.Helpers;
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
        private IRegressionModel _advModel;
        private RegModelType _advTrainedType;
        private string[] _advFeatureNames;
        private readonly List<(HyperSpec spec, Slider slider, TextBlock val)> _advHyper = new();
        private readonly List<TextBox> _advPredictBoxes = new();

        // The single Regression dropdown drives this: index 0=Linear, 1=Polynomial,
        // 2.. = advanced IRegressionModel algorithms (enum order starts at NormalEquation).
        private RegModelType AdvType() =>
            (RegModelType)Math.Max(0, CmbRegAlgo.SelectedIndex - 2);

        // Called by RegAlgo_Changed when an advanced algorithm is selected.
        private void AdvOnSelected()
        {
            if (_dataset != null && CmbAdvTarget != null && !CmbAdvTarget.Items.Cast<string>().SequenceEqual(_dataset.Headers))
            {
                CmbAdvTarget.Items.Clear();
                foreach (var h in _dataset.Headers) CmbAdvTarget.Items.Add(h);
                CmbAdvTarget.SelectedIndex = _dataset.ColCount - 1;
            }
            BuildAdvHyper();
            if (AdvTreeBorder != null)
                AdvTreeBorder.Visibility = (AdvType() == RegModelType.RandomForest) ? Visibility.Visible : Visibility.Collapsed;
        }

        private void AdvSplit_Changed(object s, RoutedPropertyChangedEventArgs<double> e)
        {
            if (TxtAdvSplit != null) { int t = (int)SliderAdvSplit.Value; TxtAdvSplit.Text = $"{t}/{100 - t}"; }
        }

        private void BuildAdvHyper()
        {
            if (AdvHyperPanel == null) return;
            AdvHyperPanel.Children.Clear();
            _advHyper.Clear();
            foreach (var spec in RegModelInfo.Hyper(AdvType()))
            {
                var val = new TextBlock { Foreground = Brushes.DarkSlateGray, FontSize = 12 };
                var lbl = new TextBlock { Foreground = AdvB("#334155"), FontSize = 12, Margin = new Thickness(0, 6, 0, 0) };
                lbl.Inlines.Add(new System.Windows.Documents.Run(spec.Name + ": "));
                lbl.Inlines.Add(val);
                var sl = new Slider { Minimum = spec.Lo, Maximum = spec.Hi, Value = spec.Default };
                if (spec.IsInteger) { sl.IsSnapToTickEnabled = true; sl.TickFrequency = 1; }
                val.Text = spec.IsInteger ? ((int)spec.Default).ToString() : spec.Default.ToString("0.####");
                sl.ValueChanged += (o, e) => val.Text = spec.IsInteger ? ((int)sl.Value).ToString() : sl.Value.ToString("0.####");
                AdvHyperPanel.Children.Add(lbl);
                AdvHyperPanel.Children.Add(sl);
                _advHyper.Add((spec, sl, val));
            }
        }

        private Dictionary<string, double> AdvReadHyper()
        {
            var d = new Dictionary<string, double>();
            foreach (var h in _advHyper)
                d[h.spec.Name] = h.spec.IsInteger ? Math.Round(h.slider.Value) : h.slider.Value;
            return d;
        }

        private class AdvResult
        {
            public CrossValidationReport Report;
            public string[] FeatureNames;
            public RegMetrics TrTrain, TrTest;
            public bool Cv; public double CvR2, CvR2Std, CvRMSE, CvMAE; public int K;
            public double[] ScAct, ScPred, ResPred, ResVal;
            public double[] LcSize, LcTrain, LcVal;
            public double[] VcVal, VcTrain, VcVal2; public string VcParam;
            public IRegressionModel FullModel; public RegTreeNode Tree;
            public int NFeatures; public string Error;
        }

        private async void AdvTrain_Click(object sender, RoutedEventArgs e)
        {
            if (_dataset == null) { TxtAdvResults.Text = "Load data first!"; return; }
            if (!CmbAdvTarget.Items.Cast<string>().SequenceEqual(_dataset.Headers))
            {
                CmbAdvTarget.Items.Clear();
                foreach (var h in _dataset.Headers) CmbAdvTarget.Items.Add(h);
                CmbAdvTarget.SelectedIndex = _dataset.ColCount - 1;
            }
            int targetCol = CmbAdvTarget.SelectedIndex;
            if (targetCol < 0) { TxtAdvResults.Text = "Select a target column."; return; }
            PredictionRecipe pipeline;
            try { pipeline = CapturePredictionRecipe(targetCol); }
            catch (System.Exception ex) { TxtAdvResults.Text = ex.Message; return; }

            var type = AdvType();
            var hyper = AdvReadHyper();
            double split = SliderAdvSplit.Value / 100.0;
            bool cv = ChkAdvKFold.IsChecked == true;
            int folds = ParseFolds(TxtAdvFolds.Text);
            bool curves = ChkAdvCurves.IsChecked == true;

            BtnAdvTrain.IsEnabled = false; AdvProgress.Visibility = Visibility.Visible; AdvProgress.IsIndeterminate = true;
            AdvCvTable.Show(null);
            TxtAdvResults.Text = "Training...";

            int resetVersion = _pageResetVersions[2];
            var trainingData = _dataset;
            AdvResult r = await Task.Run(() => AdvCompute(type, targetCol, hyper, split, cv, folds, curves, trainingData));
            if (resetVersion != _pageResetVersions[2]) { r.FullModel?.Dispose(); return; }

            if (r.Error != null)
            {
                TxtAdvResults.Text = "❌ " + r.Error;
                AdvProgress.Visibility = Visibility.Collapsed; BtnAdvTrain.IsEnabled = true;
                return;
            }

            _advModel = r.FullModel;
            _advFeatureNames = r.FeatureNames;
            _advPipeline = pipeline;
            AdvCvTable.Show(r.Report);
            _advTrainedType = AdvType();

            var sb = new StringBuilder();
            if (r.Cv)
            {
                sb.AppendLine($"══════ {r.K}-FOLD CV ══════");
                sb.AppendLine($"R²:   {r.CvR2:F4} ± {r.CvR2Std:F4}");
                sb.AppendLine($"RMSE: {r.CvRMSE:G5}");
                sb.AppendLine($"MAE:  {r.CvMAE:G5}");
                sb.AppendLine();
            }
            sb.AppendLine("══════ TRAIN ══════");
            sb.AppendLine($"R²:     {r.TrTrain.R2:F4}");
            sb.AppendLine($"AdjR²:  {r.TrTrain.AdjR2:F4}");
            sb.AppendLine($"MAE:    {r.TrTrain.MAE:G5}");
            sb.AppendLine($"RMSE:   {r.TrTrain.RMSE:G5}");
            sb.AppendLine();
            sb.AppendLine("══════ TEST ══════");
            sb.AppendLine($"R²:     {r.TrTest.R2:F4}");
            sb.AppendLine($"AdjR²:  {r.TrTest.AdjR2:F4}");
            sb.AppendLine($"MAE:    {r.TrTest.MAE:G5}");
            sb.AppendLine($"RMSE:   {r.TrTest.RMSE:G5}");
            TxtAdvResults.Text = sb.ToString();

            // record for the Compare tab
            try
            {
                string tgt = CmbAdvTarget.SelectedItem?.ToString() ?? "";
                MLRunHistory.AddRegression(RegModelInfo.Display(AdvType()), tgt, r.TrTest.R2, r.TrTest.AdjR2, r.TrTest.MAE, r.TrTest.RMSE);
            }
            catch { }

            DrawAdvScatter(r.ScAct, r.ScPred);
            DrawAdvResidual(r.ResPred, r.ResVal);
            if (curves) { DrawAdvLearning(r.LcSize, r.LcTrain, r.LcVal); DrawAdvValidation(r.VcVal, r.VcTrain, r.VcVal2, r.VcParam); }
            else { AdvLearnChart.Series = Array.Empty<ISeries>(); AdvValChart.Series = Array.Empty<ISeries>(); }

            if (type == RegModelType.RandomForest && r.Tree != null)
            {
                AdvTreeBorder.Visibility = Visibility.Visible;
                DrawAdvTree(r.Tree);
            }
            else AdvTreeBorder.Visibility = Visibility.Collapsed;

            BuildAdvPredictInputs();
            AdvProgress.Visibility = Visibility.Collapsed; BtnAdvTrain.IsEnabled = true;
        }

        // ---- all heavy work off the UI thread ----
        private AdvResult AdvCompute(RegModelType type, int targetCol, Dictionary<string, double> hyper,
            double split, bool cv, int folds, bool curves, MLDataset trainingData)
        {
            try
            {
                DatasetBuilder.Build(trainingData, targetCol, out float[][] X, out float[] y, out string[] names);
                int n = X.Length, nf = names.Length;
                var idx = DatasetBuilder.Shuffle(n);
                int trainSize = Math.Max(1, (int)(n * split));
                if (trainSize >= n) trainSize = n - 1;
                var trIdx = idx.Take(trainSize).ToArray();
                var teIdx = idx.Skip(trainSize).ToArray();
                if (teIdx.Length == 0) teIdx = trIdx;

                var trDs = DatasetBuilder.Subset(X, y, trIdx);
                var teDs = DatasetBuilder.Subset(X, y, teIdx);

                IRegressionModel Make(Dictionary<string, double> hp)
                { var m = RegModelFactory.Create(type); m.SetHyperparameters(hp); return m; }

                var res = new AdvResult { NFeatures = nf, FeatureNames = names };

                using (var splitModel = Make(hyper))
                {
                    splitModel.Train(trDs);
                    var pTr = splitModel.Predict(trDs.X);
                    var pTe = splitModel.Predict(teDs.X);
                    res.TrTrain = RegressionMetrics.Compute(trDs.y, pTr, nf);
                    res.TrTest = RegressionMetrics.Compute(teDs.y, pTe, nf);
                    res.ScAct = teDs.y.Select(v => (double)v).ToArray();
                    res.ScPred = pTe.Select(v => (double)v).ToArray();
                    res.ResPred = res.ScPred;
                    res.ResVal = new double[teDs.y.Length];
                    for (int i = 0; i < teDs.y.Length; i++) res.ResVal[i] = teDs.y[i] - pTe[i];
                }

                if (cv)
                {
                    res.Report = new CrossValidationReport();
                    if (folds < 2) folds = 2; if (folds > n) folds = n;
                    var r2 = new List<double>(); var rmse = new List<double>(); var mae = new List<double>();
                    var parts = new List<int[]>();
                    int bs = n / folds, rem = n % folds, pos = 0;
                    for (int f = 0; f < folds; f++) { int sz = bs + (f < rem ? 1 : 0); parts.Add(idx.Skip(pos).Take(sz).ToArray()); pos += sz; }
                    for (int f = 0; f < folds; f++)
                    {
                        var teI = parts[f]; var trI = idx.Where(i => !teI.Contains(i)).ToArray();
                        using (var m = Make(hyper))
                        {
                            m.Train(DatasetBuilder.Subset(X, y, trI));
                            var td = DatasetBuilder.Subset(X, y, teI);
                            var p = m.Predict(td.X);
                            var trainData = DatasetBuilder.Subset(X, y, trI);
                            res.Report.AddRegression(trainData.y.Select(v => (double)v).ToArray(),
                                m.Predict(trainData.X).Select(v => (double)v).ToArray(),
                                td.y.Select(v => (double)v).ToArray(), p.Select(v => (double)v).ToArray());
                            var mm = RegressionMetrics.Compute(td.y, p, nf);
                            r2.Add(mm.R2); rmse.Add(mm.RMSE); mae.Add(mm.MAE);
                        }
                    }
                    res.Cv = true; res.K = folds;
                    res.Report.Complete();
                    res.CvR2 = r2.Average(); res.CvRMSE = rmse.Average(); res.CvMAE = mae.Average();
                    res.CvR2Std = Math.Sqrt(r2.Select(v => (v - res.CvR2) * (v - res.CvR2)).Sum() / Math.Max(1, r2.Count));
                }

                // full-data model for predict + tree diagram
                res.FullModel = Make(hyper);
                res.FullModel.Train(new Dataset { X = X, y = y });
                if (type == RegModelType.RandomForest && res.FullModel is RandomForestModel rf) res.Tree = rf.FirstTree;

                if (curves)
                {
                    // learning curve: R² (train vs test) as training size grows
                    var sizes = new[] { 0.2, 0.4, 0.6, 0.8, 1.0 };
                    var lcS = new List<double>(); var lcT = new List<double>(); var lcV = new List<double>();
                    foreach (var frac in sizes)
                    {
                        int s = Math.Max(2, (int)(trIdx.Length * frac));
                        var sub = trIdx.Take(s).ToArray();
                        using (var m = Make(hyper))
                        {
                            m.Train(DatasetBuilder.Subset(X, y, sub));
                            var subDs = DatasetBuilder.Subset(X, y, sub);
                            var pt = m.Predict(subDs.X);
                            var pv = m.Predict(teDs.X);
                            lcS.Add(s);
                            lcT.Add(RegressionMetrics.Compute(subDs.y, pt, nf).R2);
                            lcV.Add(RegressionMetrics.Compute(teDs.y, pv, nf).R2);
                        }
                    }
                    res.LcSize = lcS.ToArray(); res.LcTrain = lcT.ToArray(); res.LcVal = lcV.ToArray();

                    // validation curve: vary the first hyperparameter
                    var specs = RegModelInfo.Hyper(type);
                    if (specs.Count > 0)
                    {
                        var sp = specs[0];
                        res.VcParam = sp.Name;
                        var vals = new List<double>(); var vt = new List<double>(); var vv = new List<double>();
                        int steps = 6;
                        for (int k = 0; k <= steps; k++)
                        {
                            double val = sp.Lo + (sp.Hi - sp.Lo) * k / steps;
                            if (sp.IsInteger) val = Math.Round(val);
                            var hp = new Dictionary<string, double>(hyper); hp[sp.Name] = val;
                            using (var m = Make(hp))
                            {
                                m.Train(trDs);
                                var pt = m.Predict(trDs.X);
                                var pv = m.Predict(teDs.X);
                                vals.Add(val);
                                vt.Add(RegressionMetrics.Compute(trDs.y, pt, nf).R2);
                                vv.Add(RegressionMetrics.Compute(teDs.y, pv, nf).R2);
                            }
                        }
                        res.VcVal = vals.ToArray(); res.VcTrain = vt.ToArray(); res.VcVal2 = vv.ToArray();
                    }
                }
                return res;
            }
            catch (Exception ex) { return new AdvResult { Error = ex.Message }; }
        }

        // ---- charts ----
        private void DrawAdvScatter(double[] act, double[] pred)
        {
            var pts = new List<ObservablePoint>();
            for (int i = 0; i < act.Length; i++) pts.Add(new ObservablePoint(act[i], pred[i]));
            double mn = act.Length > 0 ? act.Min() : 0, mx = act.Length > 0 ? act.Max() : 1;
            AdvScatter.Series = new ISeries[]
            {
                new ScatterSeries<ObservablePoint>{ Values = pts, Fill = new SolidColorPaint(SKColor.Parse("#087F73")), GeometrySize = 5, Stroke = null },
                new LineSeries<ObservablePoint>{ Values = new[]{ new ObservablePoint(mn,mn), new ObservablePoint(mx,mx)}, Fill=null, Stroke=new SolidColorPaint(SKColors.OrangeRed,2), GeometrySize=0, LineSmoothness=0 }
            };
            AdvAxes(AdvScatter, "Actual", "Predicted");
        }
        private void DrawAdvResidual(double[] pred, double[] res)
        {
            var pts = new List<ObservablePoint>();
            for (int i = 0; i < pred.Length; i++) pts.Add(new ObservablePoint(pred[i], res[i]));
            double mn = pred.Length > 0 ? pred.Min() : 0, mx = pred.Length > 0 ? pred.Max() : 1;
            AdvResidual.Series = new ISeries[]
            {
                new ScatterSeries<ObservablePoint>{ Values = pts, Fill = new SolidColorPaint(SKColor.Parse("#5275C5")), GeometrySize = 5, Stroke = null },
                new LineSeries<ObservablePoint>{ Values = new[]{ new ObservablePoint(mn,0), new ObservablePoint(mx,0)}, Fill=null, Stroke=new SolidColorPaint(SKColors.Gray,1), GeometrySize=0, LineSmoothness=0 }
            };
            AdvAxes(AdvResidual, "Predicted", "Residual");
        }
        private void DrawAdvLearning(double[] size, double[] tr, double[] val)
        {
            AdvLearnChart.Series = new ISeries[]
            {
                Line(size, tr, "#087F73", "Train"),
                Line(size, val, "#ff6b6b", "Validation")
            };
            AdvAxes(AdvLearnChart, "Train size", "R²");
        }
        private void DrawAdvValidation(double[] pv, double[] tr, double[] val, string param)
        {
            AdvValChart.Series = new ISeries[]
            {
                Line(pv, tr, "#087F73", "Train"),
                Line(pv, val, "#ff6b6b", "Validation")
            };
            AdvAxes(AdvValChart, param ?? "param", "R²");
        }
        private static LineSeries<ObservablePoint> Line(double[] x, double[] yv, string hex, string name)
        {
            var pts = new List<ObservablePoint>();
            for (int i = 0; i < x.Length; i++) pts.Add(new ObservablePoint(x[i], yv[i]));
            return new LineSeries<ObservablePoint> { Name = name, Values = pts, Fill = null, Stroke = new SolidColorPaint(SKColor.Parse(hex), 2), GeometrySize = 8, GeometryFill = new SolidColorPaint(SKColor.Parse(hex)), LineSmoothness = 0 };
        }
        private static void AdvAxes(LiveChartsCore.SkiaSharpView.WPF.CartesianChart c, string x, string y)
        {
            c.XAxes = new Axis[] { new Axis { Name = x, NamePaint = new SolidColorPaint(SKColors.Gray), LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 } };
            c.YAxes = new Axis[] { new Axis { Name = y, NamePaint = new SolidColorPaint(SKColors.Gray), LabelsPaint = new SolidColorPaint(SKColors.Gray), TextSize = 10 } };
        }

        // ---- tree branching diagram (Random Forest, first tree) ----
        private void DrawAdvTree(RegTreeNode root)
        {
            AdvTreeCanvas.Children.Clear();
            if (root == null) return;
            int maxDepth = 4;
            var dv = new DrawingVisual();
            double w = AdvTreeCanvas.ActualWidth > 10 ? AdvTreeCanvas.ActualWidth : 900;
            double h = 360;
            using (var dc = dv.RenderOpen())
            {
                void Draw(RegTreeNode n, int depth, double xmin, double xmax, double? px, double? py)
                {
                    if (n == null || depth > maxDepth) return;
                    double x = (xmin + xmax) / 2;
                    double y = 30 + depth * ((h - 50) / (maxDepth + 1));
                    if (px.HasValue)
                        dc.DrawLine(new Pen(new SolidColorBrush(Color.FromRgb(0x23, 0x35, 0x54)), 1.5), new Point(px.Value, py.Value), new Point(x, y));
                    string label = n.IsLeaf ? $"{n.Value:F2}" : $"x{n.Feature}<{n.Threshold:F2}";
                    var brush = n.IsLeaf ? new SolidColorBrush(Color.FromRgb(0x64, 0xff, 0xda)) : new SolidColorBrush(Color.FromRgb(0x0f, 0x34, 0x60));
                    var ft = MkTxt(label, 10, n.IsLeaf ? Brushes.Black : Brushes.White);
                    double bw = ft.Width + 12, bh = ft.Height + 6;
                    dc.DrawRoundedRectangle(brush, new Pen(new SolidColorBrush(Color.FromRgb(0x23, 0x35, 0x54)), 1), new Rect(x - bw / 2, y - bh / 2, bw, bh), 4, 4);
                    dc.DrawText(ft, new Point(x - ft.Width / 2, y - ft.Height / 2));
                    if (!n.IsLeaf && depth < maxDepth)
                    {
                        Draw(n.Left, depth + 1, xmin, x, x, y);
                        Draw(n.Right, depth + 1, x, xmax, x, y);
                    }
                }
                Draw(root, 0, 10, w - 10, null, null);
            }
            AdvTreeCanvas.Children.Add(new DrawingVisualHost(dv));
        }

        // ---- predict ----
        private void BuildAdvPredictInputs()
        {
            AdvPredictInputs.Children.Clear();
            _advPredictBoxes.Clear();
            if (_advFeatureNames == null) return;
            foreach (var name in _advPipeline?.InputNames ?? _advFeatureNames)
            {
                AdvPredictInputs.Children.Add(new TextBlock { Text = name, Foreground = AdvB("#334155"), FontSize = 12, Margin = new Thickness(0, 6, 0, 2) });
                var tb = new TextBox { Text = "0", FontSize = 13, Background = AdvB("#F1F5F9"), Foreground = Brushes.DarkSlateGray, BorderBrush = AdvB("#E2E7EF"), Padding = new Thickness(6, 4, 6, 4) };
                _advPredictBoxes.Add(tb);
                AdvPredictInputs.Children.Add(tb);
            }
        }
        private void AdvPredict_Click(object sender, RoutedEventArgs e)
        {
            if (_advModel == null) { AdvPredictResult.Text = "Train a model first."; return; }
            var x = new double[_advPredictBoxes.Count];
            for (int i = 0; i < x.Length; i++)
                if (!double.TryParse(_advPredictBoxes[i].Text.Trim(), NumberStyles.Any, CultureInfo.InvariantCulture, out x[i]))
                { AdvPredictResult.Text = $"Invalid value for {(_advPipeline?.InputNames ?? _advFeatureNames)[i]}"; return; }
            try
            {
                var features = (_advPipeline?.Transform(x) ?? x).Select(v => (float)v).ToArray();
                var p = _advModel.Predict(new float[][] { features });
                AdvPredictResult.Text = $"{_advPipeline?.TargetName ?? "Prediction"}: {(_advPipeline?.RestoreTarget(p[0]) ?? p[0]):G6}";
            }
            catch (Exception ex) { AdvPredictResult.Text = "Predict error: " + ex.Message; }
        }

        private void AdvSaveModel_Click(object sender, RoutedEventArgs e)
        {
            if (_advModel == null) { AdvPredictResult.Text = "Train a model first."; return; }
            var dlg = new Microsoft.Win32.SaveFileDialog
            {
                Filter = "ArchiAI Model|*.aimodel",
                FileName = RegModelInfo.Display(_advTrainedType).Replace(" ", "_") + ".aimodel"
            };
            if (dlg.ShowDialog() == true)
            {
                try { RegModelIO.Save(_advModel, _advTrainedType, _advFeatureNames, dlg.FileName, _advPipeline); AdvPredictResult.Text = "✅ Saved: " + dlg.FileName; }
                catch (Exception ex) { AdvPredictResult.Text = "Save error: " + ex.Message; }
            }
        }

        private void AdvLoadModel_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new Microsoft.Win32.OpenFileDialog { Filter = "ArchiAI Model|*.aimodel" };
            if (dlg.ShowDialog() != true) return;
            try
            {
                _advModel = RegModelIO.Load(dlg.FileName, out var type, out var feats, out _advPipeline);
                AdvCvTable.Show(null);
                _advTrainedType = type; _advFeatureNames = feats;
                CmbRegAlgo.SelectedIndex = (int)type + 2;   // switch dropdown to the matching algorithm panel
                BuildAdvPredictInputs();
                AdvPredictResult.Text = $"✅ Loaded {RegModelInfo.Display(type)} — enter feature values and Predict.";
            }
            catch (Exception ex) { AdvPredictResult.Text = "Load error: " + ex.Message; }
        }

        private static SolidColorBrush AdvB(string hex) => new SolidColorBrush((Color)ColorConverter.ConvertFromString(hex));
    }
}
