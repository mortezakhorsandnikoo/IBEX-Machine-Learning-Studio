using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace ArchiAI.ML
{
    public enum RegularizationType { L1 = 0, L2 = 1, None = 2 }

    public class LinearRegressionResult
    {
        public double[] Weights { get; set; }
        public double Bias { get; set; }
        public List<double> CostHistory { get; set; }
        public string ConvergenceMessage { get; set; }
        public double TrainR2 { get; set; }
        public double TestR2 { get; set; }
        public double TrainMAE { get; set; }
        public double TestMAE { get; set; }
        public double TrainRMSE { get; set; }
        public double TestRMSE { get; set; }
        public double[] TrainPredictions { get; set; }
        public double[] TestPredictions { get; set; }
        public double[] TrainActual { get; set; }
        public double[] TestActual { get; set; }
        public double[] Residuals { get; set; }
        public string[] FeatureNames { get; set; }

        // Normalization stats used at train time (null if Normalize was off).
        // Needed so manual predictions scale raw input the same way.
        public double[] FeatureMeans { get; set; }
        public double[] FeatureStds { get; set; }

        // Predict the target for one raw feature vector (features in column order, target excluded).
        public double Predict(double[] x)
        {
            double[] xin = x;
            if (FeatureMeans != null && FeatureStds != null)
            {
                xin = new double[x.Length];
                for (int j = 0; j < x.Length; j++)
                {
                    double s = FeatureStds[j] == 0 ? 1e-10 : FeatureStds[j];
                    xin[j] = (x[j] - FeatureMeans[j]) / s;
                }
            }
            double y = Bias;
            for (int j = 0; j < Weights.Length && j < xin.Length; j++)
                y += Weights[j] * xin[j];
            return y;
        }
    }

    // Aggregated K-Fold cross-validation result for linear regression.
    // NOTE: R2 values are in PERCENT (consistent with LinearRegressionResult.TrainR2/TestR2).
    public class LinearCVResult
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
        public LinearRegressionResult FinalModel { get; set; } // trained on all rows
    }

    public class LinearRegressionEngine
    {
        public int Iterations { get; set; } = 100;
        public double LearningRate { get; set; } = 1e-3;
        public RegularizationType Regularization { get; set; } = RegularizationType.None;
        public double RegularizationLambda { get; set; } = 0.01;
        public bool Normalize { get; set; } = false;
        public double TrainTestSplit { get; set; } = 0.8;

        private int batchSize = 32;
        private double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;

        private Vector<double> weights;
        private double bias;
        private Matrix<double> TrainX, TestX;
        private Vector<double> TrainY, TestY;

        private Vector<double> mW, vW;
        private double mB, vB;

        private double[] _normMeans, _normStds; // captured when Normalize is on

        public LinearRegressionResult Run(MLDataset dataset, int targetCol, string[] featureNames)
        {
            BuildXY(dataset, targetCol, out double[,] X, out double[] y);
            _normMeans = _normStds = null;
            if (Normalize) X = NormalizeFeatures(X);
            SplitData(X, y);
            return Train(featureNames);
        }

        // ---------------- K-FOLD CROSS-VALIDATION ----------------
        public LinearCVResult CrossValidate(MLDataset dataset, int targetCol, string[] featureNames, int kFolds)
        {
            BuildXY(dataset, targetCol, out double[,] X, out double[] y);
            _normMeans = _normStds = null;
            if (Normalize) X = NormalizeFeatures(X);

            int total = dataset.RowCount;
            int features = X.GetLength(1);
            if (kFolds < 2) kFolds = 2;
            if (kFolds > total) kFolds = total;

            int[] indices = Enumerable.Range(0, total).ToArray();
            var rand = new Random(42);
            for (int i = total - 1; i > 0; i--)
            {
                int j = rand.Next(0, i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            // Partition shuffled indices into k folds.
            var folds = new List<int[]>();
            int baseSize = total / kFolds, rem = total % kFolds, pos = 0;
            for (int f = 0; f < kFolds; f++)
            {
                int size = baseSize + (f < rem ? 1 : 0);
                folds.Add(indices.Skip(pos).Take(size).ToArray());
                pos += size;
            }

            var report = new CrossValidationReport();
            var r2 = new double[kFolds];
            var rmse = new double[kFolds];
            var mae = new double[kFolds];

            for (int f = 0; f < kFolds; f++)
            {
                int[] testIdx = folds[f];
                int[] trainIdx = indices.Where(i => !testIdx.Contains(i)).ToArray();
                SetSplit(X, y, trainIdx, testIdx, features);
                var res = Train(featureNames);
                report.AddRegression(res.TrainActual, res.TrainPredictions, res.TestActual, res.TestPredictions);
                r2[f] = res.TestR2;      // percent
                rmse[f] = res.TestRMSE;
                mae[f] = res.TestMAE;
            }

            report.Complete();
            // Final model on ALL rows (for equation/weights/charts).
            int[] all = Enumerable.Range(0, total).ToArray();
            SetSplit(X, y, all, all, features);
            var final = Train(featureNames);

            return new LinearCVResult
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

        // ---------------- shared helpers ----------------
        private void BuildXY(MLDataset dataset, int targetCol, out double[,] X, out double[] y)
        {
            int featureCount = dataset.ColCount - 1;
            X = new double[dataset.RowCount, featureCount];
            y = new double[dataset.RowCount];

            int fIdx = 0;
            for (int j = 0; j < dataset.ColCount; j++)
            {
                if (j == targetCol) continue;
                for (int i = 0; i < dataset.RowCount; i++)
                    X[i, fIdx] = dataset.Rows[i][j];
                fIdx++;
            }
            for (int i = 0; i < dataset.RowCount; i++)
                y[i] = dataset.Rows[i][targetCol];
        }

        private void SetSplit(double[,] X, double[] y, int[] trainIdx, int[] testIdx, int features)
        {
            TrainX = Matrix<double>.Build.Dense(trainIdx.Length, features, (i, j) => X[trainIdx[i], j]);
            TrainY = Vector<double>.Build.Dense(trainIdx.Length, i => y[trainIdx[i]]);
            TestX = Matrix<double>.Build.Dense(testIdx.Length, features, (i, j) => X[testIdx[i], j]);
            TestY = Vector<double>.Build.Dense(testIdx.Length, i => y[testIdx[i]]);
        }

        private static double Std(double[] v)
        {
            if (v.Length < 2) return 0;
            double m = v.Average();
            return Math.Sqrt(v.Select(x => (x - m) * (x - m)).Sum() / (v.Length - 1));
        }

        private void SplitData(double[,] X, double[] y)
        {
            int total = X.GetLength(0);
            int trainSize = (int)(total * TrainTestSplit);
            int features = X.GetLength(1);

            int[] indices = Enumerable.Range(0, total).ToArray();
            var rand = new Random(42);
            for (int i = total - 1; i > 0; i--)
            {
                int j = rand.Next(0, i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            var trainIdx = indices.Take(trainSize).ToArray();
            var testIdx = indices.Skip(trainSize).ToArray();

            TrainX = Matrix<double>.Build.Dense(trainSize, features, (i, j) => X[trainIdx[i], j]);
            TrainY = Vector<double>.Build.Dense(trainSize, i => y[trainIdx[i]]);
            TestX = Matrix<double>.Build.Dense(testIdx.Length, features, (i, j) => X[testIdx[i], j]);
            TestY = Vector<double>.Build.Dense(testIdx.Length, i => y[testIdx[i]]);
        }

        private double[,] NormalizeFeatures(double[,] X)
        {
            var matrix = Matrix<double>.Build.DenseOfArray(X);
            int rows = matrix.RowCount, cols = matrix.ColumnCount;

            var means = Vector<double>.Build.Dense(cols, j => matrix.Column(j).Average());
            var stds = Vector<double>.Build.Dense(cols, j =>
            {
                var col = matrix.Column(j);
                double mean = means[j];
                return Math.Sqrt(col.Select(v => Math.Pow(v - mean, 2)).Average());
            });

            var meanMat = Matrix<double>.Build.Dense(rows, cols, (i, j) => means[j]);
            var stdMat = Matrix<double>.Build.Dense(rows, cols, (i, j) => Math.Max(stds[j], 1e-10));

            // Remember stats so manual predictions can scale raw input identically.
            _normMeans = means.ToArray();
            _normStds = Enumerable.Range(0, cols).Select(j => Math.Max(stds[j], 1e-10)).ToArray();

            return (matrix - meanMat).PointwiseDivide(stdMat).ToArray();
        }

        private LinearRegressionResult Train(string[] featureNames)
        {
            int m = TrainX.RowCount, n = TrainX.ColumnCount;

            weights = Vector<double>.Build.Dense(n, 0.1);
            bias = 0.01;
            mW = Vector<double>.Build.Dense(n, 0);
            vW = Vector<double>.Build.Dense(n, 0);
            mB = 0; vB = 0;

            double prevCost = double.MaxValue;
            var costHistory = new List<double>();
            bool converged = false;
            int lastIter = 0;

            for (int iter = 0; iter < Iterations; iter++)
            {
                lastIter = iter;
                var batchCosts = new List<double>();

                for (int batchStart = 0; batchStart < m; batchStart += batchSize)
                {
                    int batchEnd = Math.Min(batchStart + batchSize, m);
                    var bi = Enumerable.Range(batchStart, batchEnd - batchStart).ToArray();

                    var Xb = Matrix<double>.Build.Dense(bi.Length, n, (i, j) => TrainX[bi[i], j]);
                    var Yb = Vector<double>.Build.Dense(bi.Length, i => TrainY[bi[i]]);

                    var pred = Xb * weights + bias;
                    var errors = pred - Yb;

                    double cost = errors.PointwisePower(2).Sum() / (2 * bi.Length);
                    batchCosts.Add(cost);

                    var wGrad = Xb.TransposeThisAndMultiply(errors) / bi.Length;
                    var bGrad = errors.Sum() / bi.Length;

                    if (Regularization == RegularizationType.L2)
                        wGrad += 2.0 * RegularizationLambda * weights;
                    else if (Regularization == RegularizationType.L1)
                        wGrad += RegularizationLambda * weights.Map(x => (double)Math.Sign(x));

                    // Adam optimizer
                    mW = beta1 * mW + (1 - beta1) * wGrad;
                    vW = beta2 * vW + (1 - beta2) * wGrad.PointwisePower(2);
                    mB = beta1 * mB + (1 - beta1) * bGrad;
                    vB = beta2 * vB + (1 - beta2) * bGrad * bGrad;

                    var mHat = mW / (1 - Math.Pow(beta1, iter + 1));
                    var vHat = vW / (1 - Math.Pow(beta2, iter + 1));
                    var bmHat = mB / (1 - Math.Pow(beta1, iter + 1));
                    var bvHat = vB / (1 - Math.Pow(beta2, iter + 1));

                    weights -= LearningRate * mHat.PointwiseDivide(vHat.Map(Math.Sqrt).Add(epsilon));
                    bias -= LearningRate * bmHat / (Math.Sqrt(bvHat) + epsilon);
                }

                double avgCost = batchCosts.Average();
                costHistory.Add(avgCost);

                if (Math.Abs(prevCost - avgCost) <= 1e-4) { converged = true; break; }
                prevCost = avgCost;
            }

            // Predictions
            var trainPred = (TrainX * weights + bias).ToArray();
            var testPred = (TestX * weights + bias).ToArray();
            var trainActual = TrainY.ToArray();
            var testActual = TestY.ToArray();

            // Residuals (test)
            var residuals = testActual.Zip(testPred, (a, p) => a - p).ToArray();

            return new LinearRegressionResult
            {
                Weights = weights.ToArray(),
                Bias = bias,
                CostHistory = costHistory,
                ConvergenceMessage = converged
                    ? $"Converged at iteration {lastIter + 1}, cost: {costHistory.Last():E4}"
                    : $"Did not converge after {Iterations} iterations, cost: {costHistory.Last():E4}",
                TrainR2 = R2(trainActual, trainPred),
                TestR2 = R2(testActual, testPred),
                TrainMAE = MAE(trainActual, trainPred),
                TestMAE = MAE(testActual, testPred),
                TrainRMSE = RMSE(trainActual, trainPred),
                TestRMSE = RMSE(testActual, testPred),
                TrainPredictions = trainPred,
                TestPredictions = testPred,
                TrainActual = trainActual,
                TestActual = testActual,
                Residuals = residuals,
                FeatureNames = featureNames,
                FeatureMeans = _normMeans,
                FeatureStds = _normStds
            };
        }

        private static double R2(double[] actual, double[] predicted)
        {
            double mean = actual.Average();
            double ssTotal = actual.Sum(a => (a - mean) * (a - mean));
            double ssRes = actual.Zip(predicted, (a, p) => (a - p) * (a - p)).Sum();
            return ssTotal == 0 ? 0 : (1.0 - ssRes / ssTotal) * 100;
        }

        private static double MAE(double[] actual, double[] predicted)
        {
            return actual.Zip(predicted, (a, p) => Math.Abs(a - p)).Average();
        }

        private static double RMSE(double[] actual, double[] predicted)
        {
            return Math.Sqrt(actual.Zip(predicted, (a, p) => (a - p) * (a - p)).Average());
        }
    }
}
