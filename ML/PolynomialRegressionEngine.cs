using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ArchiAI.ML
{
    // Result of a single fit (holdout, or the final all-data model under CV).
    public class PolynomialRegressionResult
    {
        public int Degree { get; set; }
        public double[] Weights { get; set; }      // raw coefficients (original units)
        public double Bias { get; set; }
        public string[] FeatureNames { get; set; } // expanded, e.g. "Area", "Area^2"
        public string TargetName { get; set; }
        public List<double> CostHistory { get; set; } = new List<double>();
        public double[] TrainPredictions { get; set; }
        public double[] TestPredictions { get; set; }
        public double[] TrainActual { get; set; }
        public double[] TestActual { get; set; }
        public double MSE { get; set; }
        public double RMSE { get; set; }
        public double MAE { get; set; }
        public double R2 { get; set; }             // on test split
        public double TrainR2 { get; set; }
        public string Equation { get; set; }

        // Predict for one raw feature vector (base features in column order, target excluded).
        // Expands to powers 1..Degree exactly as training did, then applies raw weights + bias.
        public double Predict(double[] baseFeatures)
        {
            var expanded = new List<double>();
            foreach (var v in baseFeatures)
            {
                double pow = 1.0;
                for (int p = 1; p <= Degree; p++) { pow *= v; expanded.Add(pow); }
            }
            double y = Bias;
            for (int i = 0; i < Weights.Length && i < expanded.Count; i++)
                y += Weights[i] * expanded[i];
            return y;
        }
    }

    // Aggregated K-Fold cross-validation result.
    public class PolynomialCVResult
    {
        public CrossValidationReport Report { get; set; }
        public int K { get; set; }
        public double[] FoldR2 { get; set; }
        public double[] FoldRMSE { get; set; }
        public double[] FoldMAE { get; set; }
        public double MeanR2 { get; set; }
        public double StdR2 { get; set; }
        public double MeanRMSE { get; set; }
        public double StdRMSE { get; set; }
        public double MeanMAE { get; set; }
        public double StdMAE { get; set; }
        // Model trained on ALL rows — used for the equation, weights and charts.
        public PolynomialRegressionResult FinalModel { get; set; }
    }

    public static class PolynomialRegressionEngine
    {
        // -------- Public: holdout train/test (unchanged signature) --------
        public static PolynomialRegressionResult Train(
            MLDataset ds, int targetCol, int degree,
            double learningRate, int iterations, double trainRatio, int seed = 42)
        {
            BuildFeatures(ds, targetCol, degree, out var X, out var y, out var names, out var targetName);
            int rows = y.Length;

            int[] idx = Shuffle(rows, seed);
            int trainCount = Math.Max(1, (int)Math.Round(rows * Clamp01(trainRatio)));
            if (trainCount >= rows) trainCount = rows - 1;
            int[] trainIdx = idx.Take(trainCount).ToArray();
            int[] testIdx = idx.Skip(trainCount).ToArray();

            return Fit(X, y, names, targetName, degree, trainIdx, testIdx, learningRate, iterations);
        }

        // -------- Public: K-Fold cross-validation --------
        public static PolynomialCVResult CrossValidate(
            MLDataset ds, int targetCol, int degree,
            double learningRate, int iterations, int kFolds, int seed = 42)
        {
            BuildFeatures(ds, targetCol, degree, out var X, out var y, out var names, out var targetName);
            int rows = y.Length;
            if (kFolds < 2) kFolds = 2;
            if (kFolds > rows) kFolds = rows;

            int[] idx = Shuffle(rows, seed);

            // Partition shuffled indices into k folds (sizes differ by at most 1).
            var folds = new List<int[]>();
            int baseSize = rows / kFolds, rem = rows % kFolds, pos = 0;
            for (int f = 0; f < kFolds; f++)
            {
                int size = baseSize + (f < rem ? 1 : 0);
                folds.Add(idx.Skip(pos).Take(size).ToArray());
                pos += size;
            }

            var report = new CrossValidationReport();
            var r2 = new double[kFolds];
            var rmse = new double[kFolds];
            var mae = new double[kFolds];

            for (int f = 0; f < kFolds; f++)
            {
                int[] testIdx = folds[f];
                int[] trainIdx = idx.Where(i => !testIdx.Contains(i)).ToArray();
                var res = Fit(X, y, names, targetName, degree, trainIdx, testIdx, learningRate, iterations);
                report.AddRegression(res.TrainActual, res.TrainPredictions, res.TestActual, res.TestPredictions);
                r2[f] = res.R2;
                rmse[f] = res.RMSE;
                mae[f] = res.MAE;
            }

            report.Complete();
            // Final model on ALL rows (train == test == all) for equation/weights/charts.
            int[] all = Enumerable.Range(0, rows).ToArray();
            var final = Fit(X, y, names, targetName, degree, all, all, learningRate, iterations);

            return new PolynomialCVResult
            {
                Report = report,
                K = kFolds,
                FoldR2 = r2,
                FoldRMSE = rmse,
                FoldMAE = mae,
                MeanR2 = r2.Average(),
                StdR2 = Std(r2),
                MeanRMSE = rmse.Average(),
                StdRMSE = Std(rmse),
                MeanMAE = mae.Average(),
                StdMAE = Std(mae),
                FinalModel = final
            };
        }

        // -------- Shared feature builder --------
        private static void BuildFeatures(
            MLDataset ds, int targetCol, int degree,
            out double[][] X, out double[] y, out string[] names, out string targetName)
        {
            if (ds == null || ds.RowCount < 2)
                throw new ArgumentException("Polynomial regression needs at least two numeric rows. Load and preprocess data first.");
            if (targetCol < 0 || targetCol >= ds.ColCount)
                throw new ArgumentException("Invalid target column index.");
            if (degree < 1) degree = 1;

            int cols = ds.ColCount;
            int[] featureCols = Enumerable.Range(0, cols).Where(c => c != targetCol).ToArray();
            if (featureCols.Length == 0)
                throw new ArgumentException("Need at least one feature column besides the target.");

            var nameList = new List<string>();
            foreach (int c in featureCols)
                for (int p = 1; p <= degree; p++)
                    nameList.Add(p == 1 ? ds.Headers[c] : ds.Headers[c] + "^" + p);

            int rows = ds.RowCount, nf = nameList.Count;
            X = new double[rows][];
            y = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                var rowData = ds.Rows[i];
                if (rowData.Any(v => !double.IsFinite(v)))
                    throw new ArgumentException("Data contains missing or non-finite values. Clean the dataset before training.");
                var expanded = new double[nf];
                int kk = 0;
                foreach (int c in featureCols)
                {
                    double v = rowData[c], pow = 1.0;
                    for (int p = 1; p <= degree; p++)
                    {
                        pow *= v;
                        if (!double.IsFinite(pow))
                            throw new ArgumentException("Polynomial features overflowed. Normalize your features or lower the degree.");
                        expanded[kk++] = pow;
                    }
                }
                X[i] = expanded;
                y[i] = rowData[targetCol];
            }
            names = nameList.ToArray();
            targetName = ds.Headers[targetCol];
        }

        // -------- Shared fit core (standardize on train, GD, evaluate on test) --------
        private static PolynomialRegressionResult Fit(
            double[][] X, double[] y, string[] names, string targetName, int degree,
            int[] trainIdx, int[] testIdx, double learningRate, int iterations)
        {
            if (!double.IsFinite(learningRate) || learningRate <= 0)
                throw new ArgumentException("Learning rate must be a finite number greater than zero.");
            if (iterations < 1) iterations = 1000;
            int nf = names.Length;

            // Standardize each feature using TRAIN stats.
            double[] mean = new double[nf], std = new double[nf];
            for (int f = 0; f < nf; f++)
            {
                double s = 0;
                foreach (int i in trainIdx) s += X[i][f];
                mean[f] = s / trainIdx.Length;
                double sq = 0;
                foreach (int i in trainIdx) { double d = X[i][f] - mean[f]; sq += d * d; }
                std[f] = Math.Sqrt(sq / trainIdx.Length);
                if (!double.IsFinite(mean[f]) || !double.IsFinite(std[f]))
                    throw new ArgumentException("Feature values are too large. Normalize the data or lower the degree.");
                if (std[f] < 1e-12) std[f] = 1.0;
            }

            double Sx(int i, int f) => (X[i][f] - mean[f]) / std[f];

            double[] w = new double[nf];
            double b = 0;
            var cost = new List<double>(iterations);
            int m = trainIdx.Length;

            for (int it = 0; it < iterations; it++)
            {
                double[] gW = new double[nf];
                double gB = 0, sumSq = 0;
                foreach (int i in trainIdx)
                {
                    double pred = b;
                    for (int f = 0; f < nf; f++) pred += w[f] * Sx(i, f);
                    double err = pred - y[i];
                    sumSq += err * err; gB += err;
                    for (int f = 0; f < nf; f++) gW[f] += err * Sx(i, f);
                }
                b -= learningRate * (gB / m);
                for (int f = 0; f < nf; f++) w[f] -= learningRate * (gW[f] / m);
                if (!double.IsFinite(sumSq) || !double.IsFinite(b) || w.Any(v => !double.IsFinite(v)))
                    throw new InvalidOperationException("Training diverged. Lower the learning rate or polynomial degree, then try again.");
                cost.Add(sumSq / (2.0 * m));
            }

            double Predict(int i)
            {
                double p = b;
                for (int f = 0; f < nf; f++) p += w[f] * Sx(i, f);
                return p;
            }

            double[] trainPred = trainIdx.Select(Predict).ToArray();
            double[] testPred = testIdx.Select(Predict).ToArray();
            if (trainPred.Concat(testPred).Any(v => !double.IsFinite(v)))
                throw new InvalidOperationException("Predictions overflowed. Normalize the data or lower the learning rate.");
            double[] trainAct = trainIdx.Select(i => y[i]).ToArray();
            double[] testAct = testIdx.Select(i => y[i]).ToArray();

            double mse = 0, mae = 0;
            for (int i = 0; i < testPred.Length; i++)
            {
                double e = testPred[i] - testAct[i];
                mse += e * e; mae += Math.Abs(e);
            }
            int nTest = Math.Max(1, testPred.Length);
            mse /= nTest; mae /= nTest;

            // Fold standardization into raw coefficients (original units).
            double[] rawW = new double[nf];
            double rawB = b;
            for (int f = 0; f < nf; f++) { rawW[f] = w[f] / std[f]; rawB -= w[f] * mean[f] / std[f]; }

            return new PolynomialRegressionResult
            {
                Degree = degree,
                Weights = rawW,
                Bias = rawB,
                FeatureNames = names,
                TargetName = targetName,
                CostHistory = cost,
                TrainPredictions = trainPred,
                TestPredictions = testPred,
                TrainActual = trainAct,
                TestActual = testAct,
                MSE = mse,
                RMSE = Math.Sqrt(mse),
                MAE = mae,
                R2 = R2(testAct, testPred),
                TrainR2 = R2(trainAct, trainPred),
                Equation = BuildEquation(targetName, rawB, rawW, names)
            };
        }

        // -------- helpers --------
        private static int[] Shuffle(int n, int seed)
        {
            int[] idx = Enumerable.Range(0, n).ToArray();
            var rnd = new Random(seed);
            for (int i = n - 1; i > 0; i--)
            {
                int j = rnd.Next(i + 1);
                (idx[i], idx[j]) = (idx[j], idx[i]);
            }
            return idx;
        }

        private static double Clamp01(double r) => (r <= 0 || r >= 1) ? 0.8 : r;

        private static double Std(double[] v)
        {
            if (v.Length < 2) return 0;
            double m = v.Average();
            return Math.Sqrt(v.Select(x => (x - m) * (x - m)).Sum() / (v.Length - 1));
        }

        private static double R2(double[] actual, double[] pred)
        {
            if (actual.Length == 0) return 0;
            double mean = actual.Average(), ssRes = 0, ssTot = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                double e = actual[i] - pred[i]; ssRes += e * e;
                double d = actual[i] - mean; ssTot += d * d;
            }
            return ssTot < 1e-12 ? 0 : 1.0 - ssRes / ssTot;
        }

        private static string BuildEquation(string target, double bias, double[] w, string[] names)
        {
            var sb = new StringBuilder();
            sb.Append(target).Append(" = ").Append(bias.ToString("0.####"));
            for (int f = 0; f < w.Length; f++)
            {
                double c = w[f];
                sb.Append(c >= 0 ? " + " : " - ")
                  .Append(Math.Abs(c).ToString("0.####")).Append("·").Append(names[f]);
            }
            return sb.ToString();
        }
    }
}
