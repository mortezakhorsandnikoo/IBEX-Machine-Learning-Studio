// IbexBench: headless benchmark harness for IBEX Machine Learning Studio.
// It calls the SAME engines the application uses (ArchiAI.ML.*) with the application's default
// hyperparameters, runs 5-fold cross-validation on the exported datasets, checks that training
// and prediction complete for every algorithm, and writes results_ibex.csv.
//
// Usage:   dotnet run -c Release -- <dataDir> <outputCsv>
//          (defaults: data  results_ibex.csv)
//
// AUTHOR TODO: this harness was written WITHOUT being compiled (no .NET SDK was available to the
// assistant). If the build reports errors, paste them back; they will be small signature fixes.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using ArchiAI.ML;

namespace IbexBench
{
    internal sealed class Res
    {
        public double Mean = double.NaN, Std = double.NaN, Second = double.NaN;
        public double Sil = double.NaN, Ari = double.NaN;
        public int Clusters = -1;
        public bool PredictOk;
    }

    internal static class Program
    {
        // ------------------------------------------------------------------ settings
        const int K = 5;                       // folds (same as the scikit-learn side)
        const double PolyLr = 0.05;            // AUTHOR TODO: adjust if polynomial GD diverges
        const int PolyIters = 3000;
        const double DbscanEps = 1.5;          // AUTHOR TODO: use the same eps as benchmark_sklearn.py
        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
        static StreamWriter Out;
        static readonly List<(string task, string status)> Log = new List<(string, string)>();

        // Extra, heavier configurations for the models whose slider defaults are deliberately light
        // (application defaults are still run and reported). Values stay inside the slider ranges.
        // Edit this list to try other settings; the scikit-learn side is then run with the same ones.
        static readonly (string label, RegModelType type, string cfg, (string name, double value)[] over)[] ExtraRegCfgs =
        {
            ("MLP", RegModelType.MLP, "epochs=1000;lr=0.01", new[] { ("epochs", 1000.0), ("learningRate", 0.01) }),
            ("MLP", RegModelType.MLP, "epochs=1000;lr=0.05", new[] { ("epochs", 1000.0), ("learningRate", 0.05) }),
            ("Online GD", RegModelType.OGD, "it=500;lr=0.1", new[] { ("iterations", 500.0), ("learningRate", 0.1) }),
            ("Online GD", RegModelType.OGD, "it=500;lr=0.3", new[] { ("iterations", 500.0), ("learningRate", 0.3) }),
        };

        // Extra configuration for the classifier whose slider defaults diverge on Digits (ANN).
        static readonly (string label, ClassifierType type, string cfg, int iters, double lr)[] ExtraClsCfgs =
        {
            ("ANN (MLP)", ClassifierType.ANN, "it=500;lr=0.01", 500, 0.01),
            ("ANN (MLP)", ClassifierType.ANN, "it=500;lr=0.03", 500, 0.03),
            ("ANN (MLP)", ClassifierType.ANN, "it=1000;lr=0.01", 1000, 0.01),
        };

        static readonly (string name, string file)[] RegData =
        {
            ("Diabetes", "reg_diabetes.csv"),
            ("Friedman1", "reg_friedman1.csv"),
            ("Synthetic-linear", "reg_synthetic_linear.csv"),
        };
        static readonly (string name, string file)[] ClsData =
        {
            ("Iris", "cls_iris.csv"),
            ("Wine", "cls_wine.csv"),
            ("Breast Cancer", "cls_breast_cancer.csv"),
            ("Digits", "cls_digits.csv"),
        };
        static readonly (string name, string file)[] CluData =
        {
            ("Iris", "clu_iris"),
            ("Wine", "clu_wine"),
            ("Breast Cancer", "clu_breast_cancer"),
        };

        // labels match benchmark_sklearn.py so compare_results.py can join the two CSVs
        static readonly (string label, RegModelType type)[] RegModels =
        {
            ("Normal Equation", RegModelType.NormalEquation),
            ("Locally Weighted Reg.", RegModelType.LWR),
            ("Random Forest", RegModelType.RandomForest),
            ("MLP", RegModelType.MLP),
            ("RBF surrogate", RegModelType.RBF),
            ("FastTree-like (GBM)", RegModelType.FastTree),
            ("LightGBM", RegModelType.LightGBM),
            ("Online GD", RegModelType.OGD),
            ("XGBoost", RegModelType.XGBoost),
        };
        static readonly (string label, ClassifierType type)[] ClsModels =
        {
            ("Logistic Regression (OvR)", ClassifierType.LogisticRegression),
            ("KNN (k=5)", ClassifierType.KNN),
            ("Naive Bayes", ClassifierType.NaiveBayes),
            ("SVM (linear, OvR)", ClassifierType.SVM),
            ("Random Forest", ClassifierType.RandomForest),
            ("FastTree-like (GBM)", ClassifierType.FastTree),
            ("LightGBM", ClassifierType.LightGBM),
            ("XGBoost", ClassifierType.XGBoost),
            ("ANN (MLP)", ClassifierType.ANN),
        };
        static readonly (string label, ClustererType type)[] CluModels =
        {
            ("K-Means", ClustererType.KMeans),
            ("DBSCAN", ClustererType.DBSCAN),
            ("Hierarchical (average)", ClustererType.Hierarchical),
            ("Fuzzy C-Means", ClustererType.Fuzzy),
        };

        // ------------------------------------------------------------------ main
        static int Main(string[] args)
        {
            string dataDir = args.Length > 0 ? args[0] : "data";
            string outPath = args.Length > 1 ? args[1] : "results_ibex.csv";
            if (!Directory.Exists(dataDir)) { Console.Error.WriteLine("Data folder not found: " + dataDir); return 1; }

            using (Out = new StreamWriter(outPath, false))
            {
                Out.WriteLine("task,dataset,algorithm,config,score_mean,score_std,score2,silhouette,ari,n_clusters,seconds,predict_ok,status");

                foreach (var d in RegData) RunRegression(d.name, Path.Combine(dataDir, d.file));
                foreach (var d in ClsData) RunClassification(d.name, Path.Combine(dataDir, d.file));
                foreach (var d in CluData) RunClustering(d.name, Path.Combine(dataDir, d.file + ".csv"), Path.Combine(dataDir, d.file + "_labels.csv"));
            }

            Console.WriteLine();
            Console.WriteLine("=== Functional coverage (rows = algorithm x dataset x config) ===");
            foreach (var g in Log.GroupBy(l => l.task))
                Console.WriteLine($"{g.Key}: {g.Count(x => x.status == "OK")} OK, {g.Count(x => x.status != "OK")} failed (of {g.Count()})");
            Console.WriteLine("Wrote " + Path.GetFullPath(outPath));
            return 0;
        }

        // ------------------------------------------------------------------ regression
        static void RunRegression(string name, string path)
        {
            MLDataset ds = MLDataset.LoadCsv(path);
            int target = ds.ColCount - 1;                       // exported files have the target LAST
            string[] feat = ds.Headers.Take(target).ToArray();
            double[] x0 = ds.Rows[0].Take(target).ToArray();

            // 1) Linear regression (gradient descent inside IBEX) - two configurations
            var linCfgs = new[] { ("default", 100, 1e-3), ("converged", 3000, 1e-2) };
            var linAlgs = new[]
            {
                ("Linear (no reg.)", RegularizationType.None),
                ("Linear (L1)", RegularizationType.L1),
                ("Linear (L2)", RegularizationType.L2),
            };
            foreach (var cfg in linCfgs)
                foreach (var alg in linAlgs)
                {
                    var c = cfg; var a = alg;
                    Run("regression", name, a.Item1, c.Item1, () =>
                    {
                        var eng = new LinearRegressionEngine
                        {
                            Normalize = true, Iterations = c.Item2, LearningRate = c.Item3,
                            Regularization = a.Item2, RegularizationLambda = 0.01
                        };
                        var cv = eng.CrossValidate(ds, target, feat, K);
                        var r = FromReport(cv.Report);
                        r.Second = cv.MeanRMSE;
                        r.PredictOk = IsFinite(cv.FinalModel.Predict(x0));
                        return r;
                    });
                }

            // 2) Polynomial regression (degree 2)
            Run("regression", name, "Polynomial (deg 2)", $"lr={PolyLr};it={PolyIters}", () =>
            {
                var cv = PolynomialRegressionEngine.CrossValidate(ds, target, 2, PolyLr, PolyIters, K);
                var r = FromReport(cv.Report);
                r.Second = cv.MeanRMSE;
                r.PredictOk = cv.FinalModel != null && cv.FinalModel.Weights.All(IsFinite);
                return r;
            });

            // 3) Models behind the IRegressionModel interface: same fold logic as the application
            foreach (var m in RegModels)
            {
                var mm = m;
                Run("regression", name, mm.label, "default", () => CrossValidateInterfaceModel(ds, target, mm.type));
            }

            // 4) Extra configurations (MLP and online gradient descent)
            foreach (var e in ExtraRegCfgs)
            {
                var ee = e;
                Run("regression", name, ee.label, ee.cfg, () => CrossValidateInterfaceModel(ds, target, ee.type, ee.over));
            }
        }

        // Reproduces MLView.AdvCompute: DatasetBuilder.Shuffle(n) (seed 42), contiguous folds,
        // slider-default hyperparameters. Features and target are z-scored first (like Prepare > ZScore).
        static Res CrossValidateInterfaceModel(MLDataset ds, int target, RegModelType type, (string name, double value)[] over = null)
        {
            DatasetBuilder.Build(ds, target, out float[][] X, out float[] y, out string[] names);
            double[] ys = ZScore(X, y);                         // [mean, std] of the target
            int n = X.Length, nf = names.Length;
            int[] idx = DatasetBuilder.Shuffle(n);

            var parts = new List<int[]>();
            int bs = n / K, rem = n % K, pos = 0;
            for (int f = 0; f < K; f++) { int sz = bs + (f < rem ? 1 : 0); parts.Add(idx.Skip(pos).Take(sz).ToArray()); pos += sz; }

            var hyper = RegModelInfo.Hyper(type).ToDictionary(h => h.Name, h => h.Default);
            if (over != null) foreach (var o in over) hyper[o.name] = o.value;
            var r2 = new List<double>(); var rmse = new List<double>();
            for (int f = 0; f < K; f++)
            {
                int[] te = parts[f];
                var teSet = new HashSet<int>(te);
                int[] tr = idx.Where(i => !teSet.Contains(i)).ToArray();
                using (IRegressionModel model = RegModelFactory.Create(type))
                {
                    model.SetHyperparameters(hyper);
                    model.Train(DatasetBuilder.Subset(X, y, tr));
                    Dataset td = DatasetBuilder.Subset(X, y, te);
                    float[] p = model.Predict(td.X);
                    RegMetrics mt = RegressionMetrics.Compute(td.y, p, nf);
                    r2.Add(mt.R2); rmse.Add(mt.RMSE * ys[1]);   // RMSE back in original target units
                }
            }

            var res = new Res { Mean = r2.Average(), Std = PopStd(r2), Second = rmse.Average() };
            using (IRegressionModel full = RegModelFactory.Create(type))
            {
                full.SetHyperparameters(hyper);
                full.Train(new Dataset { X = X, y = y });
                float[] p0 = full.Predict(new[] { X[0] });
                res.PredictOk = p0.Length == 1 && IsFinite(p0[0]);
            }
            return res;
        }

        // ------------------------------------------------------------------ classification
        static void RunClassification(string name, string path)
        {
            MLDataset ds = MLDataset.LoadCsv(path);
            int target = ds.ColCount - 1;
            string[] feat = ds.Headers.Take(target).ToArray();
            double[] x0 = ds.Rows[0].Take(target).ToArray();

            foreach (var m in ClsModels)
            {
                var mm = m;
                Run("classification", name, mm.label, "default", () =>
                {
                    var eng = new ClassificationEngine { Type = mm.type };   // application defaults
                    var cv = eng.CrossValidate(ds, target, feat, K);
                    double[] acc = cv.FoldAccuracy.Select(Norm01).ToArray();
                    double[] f1 = cv.FoldF1.Select(Norm01).ToArray();
                    double pred = cv.FinalModel.Predict(x0);
                    return new Res { Mean = acc.Average(), Std = PopStd(acc), Second = f1.Average(), PredictOk = IsFinite(pred) };
                });
            }

            foreach (var e in ExtraClsCfgs)
            {
                var ee = e;
                Run("classification", name, ee.label, ee.cfg, () =>
                {
                    var eng = new ClassificationEngine { Type = ee.type, Iterations = ee.iters, LearningRate = ee.lr };
                    var cv = eng.CrossValidate(ds, target, feat, K);
                    double[] acc = cv.FoldAccuracy.Select(Norm01).ToArray();
                    double[] f1 = cv.FoldF1.Select(Norm01).ToArray();
                    double pred = cv.FinalModel.Predict(x0);
                    return new Res { Mean = acc.Average(), Std = PopStd(acc), Second = f1.Average(), PredictOk = IsFinite(pred) };
                });
            }
        }

        // ------------------------------------------------------------------ clustering
        static void RunClustering(string name, string csvPath, string labelPath)
        {
            MLDataset ds = MLDataset.LoadCsv(csvPath);
            int[] truth = File.ReadAllLines(labelPath).Skip(1).Where(l => l.Trim().Length > 0)
                              .Select(l => int.Parse(l.Trim(), Inv)).ToArray();
            int k = truth.Distinct().Count();

            foreach (var m in CluModels)
            {
                var mm = m;
                Run("clustering", name, mm.label, "default", () =>
                {
                    var eng = new ClusteringEngine
                    {
                        Type = mm.type, K = k, Iterations = 100, Eps = DbscanEps, MinPts = 5,
                        Fuzziness = 2.0, Linkage = "average", Seed = 42
                    };
                    ClusterResult cr = eng.Run(ds);
                    var r = new Res { Sil = cr.Silhouette, Clusters = cr.NumClusters };
                    if (cr.Labels.Length == truth.Length) r.Ari = Ari(truth, cr.Labels);
                    int assigned = cr.Assign(ds.Rows[0]);
                    r.PredictOk = assigned >= -1;
                    return r;
                });
            }
        }

        // ------------------------------------------------------------------ helpers
        static void Run(string task, string dataset, string algo, string config, Func<Res> body)
        {
            var sw = Stopwatch.StartNew();
            string status = "OK";
            Res r = new Res();
            try { r = body(); }
            catch (Exception ex)
            {
                Exception e = ex.InnerException ?? ex;
                status = "FAILED: " + Clean(e.GetType().Name + ": " + e.Message);
            }
            sw.Stop();
            if (status == "OK" && !r.PredictOk) status = "FAILED: prediction check";
            Log.Add((task, status == "OK" ? "OK" : "FAILED"));

            Out.WriteLine(string.Join(",", new[]
            {
                task, Clean(dataset), Clean(algo), Clean(config),
                F(r.Mean), F(r.Std), F(r.Second), F(r.Sil), F(r.Ari),
                r.Clusters.ToString(Inv), sw.Elapsed.TotalSeconds.ToString("F3", Inv),
                r.PredictOk ? "yes" : "no", status
            }));
            Out.Flush();
            Console.WriteLine($"{task,-14} {dataset,-16} {algo,-26} {config,-12} {status}  ({sw.Elapsed.TotalSeconds:F2}s)");
        }

        static Res FromReport(CrossValidationReport rep)
        {
            double[] folds = rep.Rows.Where(r => r.Fold != "Combined").Select(r => r.TestScore).ToArray();
            return new Res { Mean = folds.Average(), Std = PopStd(folds) };
        }

        // z-scores X columns and y in place (sample std, like MathNet); returns {yMean, yStd}
        static double[] ZScore(float[][] X, float[] y)
        {
            int n = X.Length, nf = X[0].Length;
            for (int j = 0; j < nf; j++)
            {
                double mu = 0; for (int i = 0; i < n; i++) mu += X[i][j]; mu /= n;
                double s = 0; for (int i = 0; i < n; i++) { double d = X[i][j] - mu; s += d * d; }
                s = Math.Sqrt(s / Math.Max(1, n - 1)); if (s < 1e-12) s = 1.0;
                for (int i = 0; i < n; i++) X[i][j] = (float)((X[i][j] - mu) / s);
            }
            double ym = 0; for (int i = 0; i < n; i++) ym += y[i]; ym /= n;
            double ys = 0; for (int i = 0; i < n; i++) { double d = y[i] - ym; ys += d * d; }
            ys = Math.Sqrt(ys / Math.Max(1, n - 1)); if (ys < 1e-12) ys = 1.0;
            for (int i = 0; i < n; i++) y[i] = (float)((y[i] - ym) / ys);
            return new[] { ym, ys };
        }

        static double Ari(int[] a, int[] b)
        {
            int n = a.Length;
            int[] ua = a.Distinct().ToArray(), ub = b.Distinct().ToArray();
            var t = new long[ua.Length, ub.Length];
            for (int i = 0; i < n; i++) t[Array.IndexOf(ua, a[i]), Array.IndexOf(ub, b[i])]++;
            double C2(long x) => x * (x - 1) / 2.0;
            double sumIj = 0, sumA = 0, sumB = 0;
            for (int i = 0; i < ua.Length; i++)
            {
                long rs = 0;
                for (int j = 0; j < ub.Length; j++) { sumIj += C2(t[i, j]); rs += t[i, j]; }
                sumA += C2(rs);
            }
            for (int j = 0; j < ub.Length; j++)
            {
                long cs = 0; for (int i = 0; i < ua.Length; i++) cs += t[i, j];
                sumB += C2(cs);
            }
            double expected = sumA * sumB / C2(n);
            double max = 0.5 * (sumA + sumB);
            return Math.Abs(max - expected) < 1e-12 ? 0.0 : (sumIj - expected) / (max - expected);
        }

        static double PopStd(IEnumerable<double> v)
        {
            double[] a = v.ToArray();
            if (a.Length == 0) return double.NaN;
            double m = a.Average();
            return Math.Sqrt(a.Sum(x => (x - m) * (x - m)) / a.Length);
        }

        // some engines may report percentages; the paper needs fractions
        static double Norm01(double v) => v > 1.0000001 ? v / 100.0 : v;
        static bool IsFinite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);
        static bool IsFinite(float v) => !float.IsNaN(v) && !float.IsInfinity(v);
        static string F(double v) => IsFinite(v) ? v.ToString("G8", Inv) : "";
        static string Clean(string s) => (s ?? "").Replace(',', ';').Replace('\r', ' ').Replace('\n', ' ').Replace('"', '\'');
    }
}
