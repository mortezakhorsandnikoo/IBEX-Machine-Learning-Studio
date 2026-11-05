using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ArchiAI.ML
{
    public enum ClassifierType { LogisticRegression = 0, KNN = 1, NaiveBayes = 2, SVM = 3, RandomForest = 4, FastTree = 5, LightGBM = 6, XGBoost = 7, ANN = 8 }

    public class ClassificationResult
    {
        public PredictionRecipe Pipeline { get; set; }
        public string TargetName { get; set; }
        public string[] InputNames => Pipeline?.InputNames ?? FeatureNames;
        public double PredictRaw(double[] raw) => Pipeline == null ? Predict(raw) : Pipeline.RestoreTarget(Predict(Pipeline.Transform(raw)));
        public ClassifierType Type { get; set; }
        public string[] FeatureNames { get; set; }
        public double[] Classes { get; set; }          // distinct class labels (sorted)

        public double TrainAccuracy { get; set; }
        public double TestAccuracy { get; set; }
        public double[] Precision { get; set; }        // per class (test)
        public double[] Recall { get; set; }
        public double[] F1 { get; set; }
        public double MacroF1 { get; set; }
        public int[,] Confusion { get; set; }          // [actualClassIdx, predClassIdx] on test
        public double[] TestActual { get; set; }
        public double[] TestPredictions { get; set; }
        public string ConvergenceMessage { get; set; }

        // ---- model (for manual predict) ----
        public double[] FeatureMeans { get; set; }
        public double[] FeatureStds { get; set; }
        public double[][] LogRegWeights { get; set; }  // [nClasses][nFeat+1], bias at last index
        public double[][] KnnX { get; set; }           // standardized training features
        public int[] KnnY { get; set; }                // class index per training row
        public int K { get; set; }

        // Trained classifier operating on STANDARDIZED features -> class index.
        // Set for every algorithm; used by Predict (in-memory model).
        public Func<double[], int> ClassifyStd { get; set; }

        // Serializable parameters (for Save/Load).
        public ClassifierState Model { get; set; }

        private double[] Std(double[] x)
        {
            var xin = new double[x.Length];
            for (int j = 0; j < x.Length; j++)
            {
                double s = (FeatureStds == null || FeatureStds[j] == 0) ? 1e-10 : FeatureStds[j];
                double mu = FeatureMeans == null ? 0 : FeatureMeans[j];
                xin[j] = (x[j] - mu) / s;
            }
            return xin;
        }

        // Predict class label for one raw feature vector (target excluded, column order).
        public double Predict(double[] rawX)
        {
            var x = Std(rawX);
            int idx;
            if (ClassifyStd != null) idx = ClassifyStd(x);
            else if (Type == ClassifierType.LogisticRegression)
            {
                double best = double.NegativeInfinity; idx = 0;
                for (int c = 0; c < LogRegWeights.Length; c++)
                {
                    var w = LogRegWeights[c];
                    double z = w[w.Length - 1];
                    for (int j = 0; j < x.Length && j < w.Length - 1; j++) z += w[j] * x[j];
                    double p = 1.0 / (1.0 + Math.Exp(-z));
                    if (p > best) { best = p; idx = c; }
                }
            }
            else idx = KnnVote(x);
            return Classes[idx];
        }

        private int KnnVote(double[] x)
        {
            var dists = new (double d, int cls)[KnnX.Length];
            for (int i = 0; i < KnnX.Length; i++)
            {
                double s = 0;
                for (int j = 0; j < x.Length; j++) { double diff = x[j] - KnnX[i][j]; s += diff * diff; }
                dists[i] = (s, KnnY[i]);
            }
            var top = dists.OrderBy(t => t.d).Take(Math.Max(1, K));
            return top.GroupBy(t => t.cls).OrderByDescending(g => g.Count())
                      .ThenBy(g => g.Key).First().Key;
        }
    }

    public class ClassificationCVResult
    {
        public CrossValidationReport Report { get; set; }
        public int K { get; set; }
        public double[] FoldAccuracy { get; set; }
        public double[] FoldF1 { get; set; }
        public double MeanAccuracy { get; set; }
        public double StdAccuracy { get; set; }
        public double MeanF1 { get; set; }
        public double StdF1 { get; set; }
        public ClassificationResult FinalModel { get; set; }
    }

    public class ClassificationEngine
    {
        public ClassifierType Type { get; set; } = ClassifierType.LogisticRegression;
        public int Iterations { get; set; } = 500;
        public double LearningRate { get; set; } = 0.1;
        public int Neighbors { get; set; } = 5;
        public double TrainTestSplit { get; set; } = 0.8;
        public double SvmC { get; set; } = 1.0;         // SVM regularization strength
        public int Trees { get; set; } = 50;            // Random Forest / boosted
        public int MaxDepth { get; set; } = 12;
        public int MinLeaf { get; set; } = 5;
        public int Seed { get; set; } = 42;
        public int AnnHidden { get; set; } = 32;        // ANN
        public int AnnLayers { get; set; } = 1;

        public ClassificationResult Run(MLDataset dataset, int targetCol, string[] featureNames)
        {
            Build(dataset, targetCol, out double[][] X, out int[] yIdx, out double[] classes, out double[] means, out double[] stds);
            int total = X.Length;
            var idx = Shuffle(total);
            int trainSize = Math.Max(1, (int)(total * TrainTestSplit));
            var trainIdx = idx.Take(trainSize).ToArray();
            var testIdx = idx.Skip(trainSize).ToArray();
            if (testIdx.Length == 0) testIdx = trainIdx;
            return BuildResult(X, yIdx, classes, means, stds, trainIdx, testIdx, featureNames);
        }

        public ClassificationCVResult CrossValidate(MLDataset dataset, int targetCol, string[] featureNames, int kFolds)
        {
            Build(dataset, targetCol, out double[][] X, out int[] yIdx, out double[] classes, out double[] means, out double[] stds);
            int total = X.Length;
            if (kFolds < 2) kFolds = 2;
            if (kFolds > total) kFolds = total;
            var idx = Shuffle(total);
            var folds = new List<int[]>();
            int baseSize = total / kFolds, rem = total % kFolds, pos = 0;
            for (int f = 0; f < kFolds; f++)
            {
                int size = baseSize + (f < rem ? 1 : 0);
                folds.Add(idx.Skip(pos).Take(size).ToArray());
                pos += size;
            }
            var report = new CrossValidationReport(true);
            var acc = new double[kFolds];
            var f1 = new double[kFolds];
            for (int f = 0; f < kFolds; f++)
            {
                var testIdx = folds[f];
                var trainIdx = idx.Where(i => !testIdx.Contains(i)).ToArray();
                var r = BuildResult(X, yIdx, classes, means, stds, trainIdx, testIdx, featureNames);
                report.AddClassification(trainIdx.Length, testIdx.Length, r.TrainAccuracy, r.TestAccuracy);
                acc[f] = r.TestAccuracy;
                f1[f] = r.MacroF1;
            }
            report.Complete();
            var all = Enumerable.Range(0, total).ToArray();
            var final = BuildResult(X, yIdx, classes, means, stds, all, all, featureNames);
            return new ClassificationCVResult
            {
                Report = report,
                K = kFolds,
                FoldAccuracy = acc,
                FoldF1 = f1,
                MeanAccuracy = acc.Average(),
                StdAccuracy = Std(acc),
                MeanF1 = f1.Average(),
                StdF1 = Std(f1),
                FinalModel = final
            };
        }

        // ---------------- core ----------------
        private ClassificationResult BuildResult(double[][] X, int[] yIdx, double[] classes,
            double[] means, double[] stds, int[] trainIdx, int[] testIdx, string[] featureNames)
        {
            int nFeat = X[0].Length;
            int nClasses = classes.Length;
            var Xtr = trainIdx.Select(i => X[i]).ToArray();
            var ytr = trainIdx.Select(i => yIdx[i]).ToArray();
            var Xte = testIdx.Select(i => X[i]).ToArray();
            var yte = testIdx.Select(i => yIdx[i]).ToArray();

            var res = new ClassificationResult
            {
                Type = Type,
                FeatureNames = featureNames,
                Classes = classes,
                FeatureMeans = means,
                FeatureStds = stds,
                K = Neighbors
            };

            var st = new ClassifierState
            {
                Type = (int)Type, Classes = classes, Means = means, Stds = stds,
                NFeat = nFeat, NClasses = nClasses, K = Neighbors
            };
            switch (Type)
            {
                case ClassifierType.LogisticRegression:
                {
                    var W = new double[nClasses][];
                    for (int c = 0; c < nClasses; c++) W[c] = TrainBinaryLogistic(Xtr, ytr.Select(v => v == c ? 1.0 : 0.0).ToArray(), nFeat);
                    st.LogW = W; res.LogRegWeights = W;
                    res.ConvergenceMessage = $"Logistic regression (one-vs-rest), {Iterations} iters";
                    break;
                }
                case ClassifierType.KNN:
                    st.KnnX = Xtr; st.KnnY = ytr; res.KnnX = Xtr; res.KnnY = ytr;
                    res.ConvergenceMessage = $"KNN, k={Neighbors}";
                    break;
                case ClassifierType.NaiveBayes:
                {
                    var mean = new double[nClasses][]; var var_ = new double[nClasses][]; var prior = new double[nClasses];
                    for (int c = 0; c < nClasses; c++) { mean[c] = new double[nFeat]; var_[c] = new double[nFeat]; }
                    var cnt = new int[nClasses];
                    for (int i = 0; i < Xtr.Length; i++) { cnt[ytr[i]]++; for (int j = 0; j < nFeat; j++) mean[ytr[i]][j] += Xtr[i][j]; }
                    for (int c = 0; c < nClasses; c++) for (int j = 0; j < nFeat; j++) mean[c][j] /= Math.Max(1, cnt[c]);
                    for (int i = 0; i < Xtr.Length; i++) for (int j = 0; j < nFeat; j++) { double d = Xtr[i][j] - mean[ytr[i]][j]; var_[ytr[i]][j] += d * d; }
                    for (int c = 0; c < nClasses; c++) { for (int j = 0; j < nFeat; j++) var_[c][j] = var_[c][j] / Math.Max(1, cnt[c]) + 1e-6; prior[c] = (double)cnt[c] / Xtr.Length; }
                    st.NbMean = mean; st.NbVar = var_; st.NbPrior = prior;
                    res.ConvergenceMessage = "Gaussian Naive Bayes";
                    break;
                }
                case ClassifierType.SVM:
                {
                    var W = new double[nClasses][];
                    for (int c = 0; c < nClasses; c++) W[c] = TrainLinearSvm(Xtr, ytr.Select(v => v == c ? 1.0 : -1.0).ToArray(), nFeat);
                    st.SvmW = W;
                    res.ConvergenceMessage = $"Linear SVM (one-vs-rest), C={SvmC}";
                    break;
                }
                case ClassifierType.FastTree:
                    st.MlnetB64 = TrainMlNetBytes(Xtr, ytr, nFeat, nClasses, lightgbm: false); st.Lightgbm = false;
                    res.ConvergenceMessage = $"FastTree multiclass (one-vs-all), {Trees} trees";
                    break;
                case ClassifierType.LightGBM:
                    st.MlnetB64 = TrainMlNetBytes(Xtr, ytr, nFeat, nClasses, lightgbm: true); st.Lightgbm = true;
                    res.ConvergenceMessage = $"LightGBM multiclass, {Trees} iters";
                    break;
                case ClassifierType.XGBoost:
                    st.XgbB64 = TrainXgbBytes(Xtr, ytr, nFeat, nClasses);
                    res.ConvergenceMessage = $"XGBoost multiclass (softmax), {Trees} rounds";
                    break;
                case ClassifierType.ANN:
                    TrainAnnSoftmax(Xtr, ytr, nFeat, nClasses, st);
                    res.ConvergenceMessage = $"Neural Net (softmax), {AnnLayers}x{AnnHidden}";
                    break;
                default: // RandomForest
                    st.Forest = TrainForest(Xtr, ytr, nClasses, nFeat).Select(ToDTO).ToArray();
                    res.ConvergenceMessage = $"Random Forest, {Trees} trees";
                    break;
            }
            res.Model = st;
            var classify = ClassifierRebuild.Build(st);
            res.ClassifyStd = classify;

            // predictions
            var trPred = Xtr.Select(classify).ToArray();
            var tePred = Xte.Select(classify).ToArray();

            res.TrainAccuracy = ytr.Zip(trPred, (a, p) => a == p ? 1 : 0).Average() * 100;
            res.TestAccuracy = yte.Zip(tePred, (a, p) => a == p ? 1 : 0).Average() * 100;

            var conf = new int[nClasses, nClasses];
            for (int i = 0; i < yte.Length; i++) conf[yte[i], tePred[i]]++;
            res.Confusion = conf;

            var prec = new double[nClasses];
            var rec = new double[nClasses];
            var f1 = new double[nClasses];
            for (int c = 0; c < nClasses; c++)
            {
                int tp = conf[c, c];
                int fp = 0, fn = 0;
                for (int k = 0; k < nClasses; k++)
                {
                    if (k != c) { fp += conf[k, c]; fn += conf[c, k]; }
                }
                prec[c] = (tp + fp) == 0 ? 0 : (double)tp / (tp + fp);
                rec[c] = (tp + fn) == 0 ? 0 : (double)tp / (tp + fn);
                f1[c] = (prec[c] + rec[c]) == 0 ? 0 : 2 * prec[c] * rec[c] / (prec[c] + rec[c]);
            }
            res.Precision = prec; res.Recall = rec; res.F1 = f1;
            res.MacroF1 = f1.Average();
            res.TestActual = yte.Select(i => classes[i]).ToArray();
            res.TestPredictions = tePred.Select(i => classes[i]).ToArray();
            return res;
        }

        private double[] TrainBinaryLogistic(double[][] X, double[] y, int nFeat)
        {
            var w = new double[nFeat];
            double b = 0;
            int m = X.Length;
            for (int iter = 0; iter < Iterations; iter++)
            {
                var gw = new double[nFeat];
                double gb = 0;
                for (int i = 0; i < m; i++)
                {
                    double z = b;
                    for (int j = 0; j < nFeat; j++) z += w[j] * X[i][j];
                    double p = 1.0 / (1.0 + Math.Exp(-z));
                    double err = p - y[i];
                    for (int j = 0; j < nFeat; j++) gw[j] += err * X[i][j];
                    gb += err;
                }
                for (int j = 0; j < nFeat; j++) w[j] -= LearningRate * gw[j] / m;
                b -= LearningRate * gb / m;
            }
            var outw = new double[nFeat + 1];
            Array.Copy(w, outw, nFeat);
            outw[nFeat] = b;
            return outw;
        }

        // ---- linear SVM (one binary, y in {+1,-1}) ----
        private double[] TrainLinearSvm(double[][] X, double[] y, int nFeat)
        {
            var w = new double[nFeat]; double b = 0;
            int m = Math.Max(1, X.Length);
            double lambda = 1.0 / (Math.Max(1e-6, SvmC) * m);
            for (int iter = 0; iter < Iterations; iter++)
                for (int i = 0; i < X.Length; i++)
                {
                    double z = b; for (int j = 0; j < nFeat; j++) z += w[j] * X[i][j];
                    double margin = y[i] * z;
                    if (margin < 1)
                    {
                        for (int j = 0; j < nFeat; j++) w[j] += LearningRate * (y[i] * X[i][j] - lambda * w[j]);
                        b += LearningRate * y[i];
                    }
                    else for (int j = 0; j < nFeat; j++) w[j] -= LearningRate * lambda * w[j];
                }
            var outw = new double[nFeat + 1]; Array.Copy(w, outw, nFeat); outw[nFeat] = b; return outw;
        }

        // ---- classification decision tree + random forest ----
        private class ClsNode { public bool IsLeaf; public int ClassIdx; public int Feature; public double Threshold; public ClsNode Left, Right; }

        private ClsNode[] TrainForest(double[][] X, int[] y, int nClasses, int nFeat)
        {
            int md = MaxDepth <= 0 ? int.MaxValue : MaxDepth;
            var forest = new ClsNode[Math.Max(1, Trees)];
            System.Threading.Tasks.Parallel.For(0, forest.Length, t =>
            {
                var rng = new Random(Seed + t * 9973);
                int n = X.Length;
                var bx = new double[n][]; var by = new int[n];
                for (int i = 0; i < n; i++) { int id = rng.Next(n); bx[i] = X[id]; by[i] = y[id]; }
                forest[t] = BuildClsTree(bx, by, nClasses, nFeat, 0, md, rng);
            });
            return forest;
        }
        private ClsNode BuildClsTree(double[][] X, int[] y, int nClasses, int nFeat, int depth, int maxDepth, Random rng)
        {
            int Majority()
            { var c = new int[nClasses]; foreach (var v in y) c[v]++; int b = 0; for (int i = 1; i < nClasses; i++) if (c[i] > c[b]) b = i; return b; }
            if (depth >= maxDepth || y.Length <= MinLeaf || y.Distinct().Count() == 1)
                return new ClsNode { IsLeaf = true, ClassIdx = y.Length > 0 ? Majority() : 0 };
            int k = Math.Max(1, (int)Math.Sqrt(nFeat));
            var feats = Enumerable.Range(0, nFeat).OrderBy(_ => rng.Next()).Take(k);
            int bestF = -1; double bestT = 0, bestGini = double.MaxValue; bool found = false;
            foreach (int f in feats)
            {
                var vals = X.Select(r => r[f]).OrderBy(v => v).ToArray();
                int step = Math.Max(1, vals.Length / 20);
                for (int i = 0; i < vals.Length - 1; i += step)
                {
                    double thr = 0.5 * (vals[i] + vals[i + 1]);
                    var lc = new int[nClasses]; var rc = new int[nClasses]; int ln = 0, rn = 0;
                    for (int r = 0; r < X.Length; r++) { if (X[r][f] < thr) { lc[y[r]]++; ln++; } else { rc[y[r]]++; rn++; } }
                    if (ln < MinLeaf || rn < MinLeaf) continue;
                    double gini = (ln * Gini(lc, ln) + rn * Gini(rc, rn)) / (ln + rn);
                    if (gini < bestGini) { bestGini = gini; bestF = f; bestT = thr; found = true; }
                }
            }
            if (!found) return new ClsNode { IsLeaf = true, ClassIdx = Majority() };
            var lx = new List<double[]>(); var ly = new List<int>(); var rx = new List<double[]>(); var ry = new List<int>();
            for (int r = 0; r < X.Length; r++) { if (X[r][bestF] < bestT) { lx.Add(X[r]); ly.Add(y[r]); } else { rx.Add(X[r]); ry.Add(y[r]); } }
            return new ClsNode
            {
                IsLeaf = false, Feature = bestF, Threshold = bestT,
                Left = BuildClsTree(lx.ToArray(), ly.ToArray(), nClasses, nFeat, depth + 1, maxDepth, rng),
                Right = BuildClsTree(rx.ToArray(), ry.ToArray(), nClasses, nFeat, depth + 1, maxDepth, rng)
            };
        }
        private static double Gini(int[] counts, int n) { if (n == 0) return 0; double g = 1; foreach (var c in counts) { double p = (double)c / n; g -= p * p; } return g; }
        private static int PredictTree(ClsNode n, double[] x) { while (!n.IsLeaf) n = x[n.Feature] < n.Threshold ? n.Left : n.Right; return n.ClassIdx; }
        private static ClsTreeDTO ToDTO(ClsNode n) => n == null ? null : new ClsTreeDTO { IsLeaf = n.IsLeaf, ClassIdx = n.ClassIdx, Feature = n.Feature, Threshold = n.Threshold, Left = ToDTO(n.Left), Right = ToDTO(n.Right) };

        // ---- ML.NET multiclass: train, return the saved model as base64 (rebuilt for predict in ClassifierRebuild) ----
        private string TrainMlNetBytes(double[][] Xtr, int[] ytr, int nFeat, int nClasses, bool lightgbm)
        {
            var ml = new MLContext(seed: Seed);
            var rows = new List<McInput>(Xtr.Length);
            for (int i = 0; i < Xtr.Length; i++)
                rows.Add(new McInput { Features = Xtr[i].Select(v => (float)v).ToArray(), Label = ytr[i] });
            var sd = SchemaDefinition.Create(typeof(McInput));
            sd["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, nFeat);
            var dv = ml.Data.LoadFromEnumerable(rows, sd);

            IEstimator<ITransformer> trainer = lightgbm
                ? ml.MulticlassClassification.Trainers.LightGbm(labelColumnName: "LabelKey", featureColumnName: "Features")
                : ml.MulticlassClassification.Trainers.OneVersusAll(
                    ml.BinaryClassification.Trainers.FastTree(
                        labelColumnName: "LabelKey",
                        numberOfLeaves: Math.Max(2, MaxDepth * 2), numberOfTrees: Trees,
                        minimumExampleCountPerLeaf: MinLeaf, learningRate: LearningRate),
                    labelColumnName: "LabelKey");

            var pipe = ml.Transforms.Conversion.MapValueToKey("LabelKey", "Label")
                .Append(trainer)
                .Append(ml.Transforms.Conversion.MapKeyToValue("PredLabel", "PredictedLabel"));
            var model = pipe.Fit(dv);
            using var ms = new System.IO.MemoryStream();
            ml.Model.Save(model, dv.Schema, ms);
            return Convert.ToBase64String(ms.ToArray());
        }

        // ---- XGBoost multiclass: train via high-level XGBClassifier, return model file bytes as base64 ----
        private string TrainXgbBytes(double[][] Xtr, int[] ytr, int nFeat, int nClasses)
        {
            var X = Xtr.Select(r => r.Select(v => (float)v).ToArray()).ToArray();
            var y = ytr.Select(v => (float)v).ToArray();
            var clf = new XGBoostSharp.XGBClassifier(maxDepth: MaxDepth, learningRate: (float)LearningRate,
                nEstimators: Trees, objective: "multi:softmax", numClass: Math.Max(2, nClasses));
            clf.Fit(X, y);
            var tmp = System.IO.Path.GetTempFileName();
            try { clf.SaveModelToFile(tmp); return Convert.ToBase64String(System.IO.File.ReadAllBytes(tmp)); }
            finally { try { System.IO.File.Delete(tmp); } catch { } clf.Dispose(); }
        }

        // ---- softmax neural net (pure C#): trains and stores weights into the state ----
        private void TrainAnnSoftmax(double[][] X, int[] y, int nFeat, int nClasses, ClassifierState st)
        {
            var rnd = new Random(Seed);
            int layers = Math.Max(1, AnnLayers) + 1;   // hidden layers + output
            int hid = Math.Max(2, AnnHidden);
            var W = new double[layers][][]; var b = new double[layers][];
            for (int l = 0; l < layers; l++)
            {
                int inS = l == 0 ? nFeat : hid;
                int outS = l == layers - 1 ? nClasses : hid;
                W[l] = new double[inS][]; for (int i = 0; i < inS; i++) W[l][i] = new double[outS];
                b[l] = new double[outS];
                double sd = Math.Sqrt(2.0 / (inS + outS));
                for (int i = 0; i < inS; i++) for (int j = 0; j < outS; j++)
                { double u1 = 1 - rnd.NextDouble(), u2 = 1 - rnd.NextDouble(); W[l][i][j] = sd * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2); }
            }
            double Relu(double z) => z > 0 ? z : 0;
            double[] Forward(double[] x, out double[][] acts, out double[][] zs)
            {
                acts = new double[layers + 1][]; zs = new double[layers][];
                acts[0] = x;
                for (int l = 0; l < layers; l++)
                {
                    int inS = W[l].Length, outS = W[l][0].Length;
                    var z = new double[outS];
                    for (int j = 0; j < outS; j++) { double s = b[l][j]; for (int k = 0; k < inS; k++) s += acts[l][k] * W[l][k][j]; z[j] = s; }
                    zs[l] = z;
                    var a = new double[outS];
                    if (l == layers - 1) { double mx = z.Max(), sum = 0; for (int j = 0; j < outS; j++) { a[j] = Math.Exp(z[j] - mx); sum += a[j]; } for (int j = 0; j < outS; j++) a[j] /= sum; }
                    else for (int j = 0; j < outS; j++) a[j] = Relu(z[j]);
                    acts[l + 1] = a;
                }
                return acts[layers];
            }
            int n = X.Length;
            for (int ep = 0; ep < Iterations; ep++)
                for (int i = 0; i < n; i++)
                {
                    var outp = Forward(X[i], out var acts, out var zs);
                    var dZ = new double[nClasses];
                    for (int j = 0; j < nClasses; j++) dZ[j] = outp[j] - (y[i] == j ? 1.0 : 0.0);
                    for (int l = layers - 1; l >= 0; l--)
                    {
                        int inS = W[l].Length, outS = W[l][0].Length;
                        var aprev = acts[l];
                        var dPrev = new double[inS];
                        for (int k = 0; k < inS; k++)
                            for (int j = 0; j < outS; j++) dPrev[k] += dZ[j] * W[l][k][j];
                        for (int k = 0; k < inS; k++)
                            for (int j = 0; j < outS; j++) W[l][k][j] -= LearningRate * dZ[j] * aprev[k];
                        for (int j = 0; j < outS; j++) b[l][j] -= LearningRate * dZ[j];
                        if (l > 0) { var nd = new double[inS]; for (int k = 0; k < inS; k++) nd[k] = dPrev[k] * (zs[l - 1][k] > 0 ? 1 : 0); dZ = nd; }
                    }
                }
            st.AnnW = W; st.AnnB = b; st.AnnTotal = layers;
        }

        // features (standardized) + class index per row
        private void Build(MLDataset dataset, int targetCol, out double[][] X, out int[] yIdx,
            out double[] classes, out double[] means, out double[] stds)
        {
            int rows = dataset.RowCount;
            int nFeat = dataset.ColCount - 1;
            var raw = new double[rows][];
            var yVals = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                raw[i] = new double[nFeat];
                int f = 0;
                for (int j = 0; j < dataset.ColCount; j++)
                {
                    if (j == targetCol) { yVals[i] = dataset.Rows[i][j]; continue; }
                    raw[i][f++] = dataset.Rows[i][j];
                }
            }
            // standardize features (full-data stats)
            means = new double[nFeat]; stds = new double[nFeat];
            for (int j = 0; j < nFeat; j++)
            {
                double mu = 0; for (int i = 0; i < rows; i++) mu += raw[i][j]; mu /= rows;
                double s = 0; for (int i = 0; i < rows; i++) { double d = raw[i][j] - mu; s += d * d; }
                s = Math.Sqrt(s / rows); if (s == 0) s = 1e-10;
                means[j] = mu; stds[j] = s;
            }
            X = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                X[i] = new double[nFeat];
                for (int j = 0; j < nFeat; j++) X[i][j] = (raw[i][j] - means[j]) / stds[j];
            }
            classes = yVals.Distinct().OrderBy(v => v).ToArray();
            var map = new Dictionary<double, int>();
            for (int c = 0; c < classes.Length; c++) map[classes[c]] = c;
            yIdx = yVals.Select(v => map[v]).ToArray();
        }

        private static int[] Shuffle(int total)
        {
            var idx = Enumerable.Range(0, total).ToArray();
            var rand = new Random(42);
            for (int i = total - 1; i > 0; i--)
            {
                int j = rand.Next(0, i + 1);
                (idx[i], idx[j]) = (idx[j], idx[i]);
            }
            return idx;
        }

        private static double Std(double[] v)
        {
            if (v.Length < 2) return 0;
            double m = v.Average();
            return Math.Sqrt(v.Select(x => (x - m) * (x - m)).Sum() / (v.Length - 1));
        }
    }
}
