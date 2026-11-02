using System;
using System.Collections.Generic;
using System.Linq;

namespace ArchiAI.ML
{
    public sealed class CrossValidationRow
    {
        public string Fold { get; set; }
        public int TrainRows { get; set; }
        public int TestRows { get; set; }
        public double TrainScore { get; set; }
        public double TestScore { get; set; }
    }

    // Combines predictions, not fold R² values. Each test observation occurs exactly once.
    public sealed class CrossValidationReport
    {
        public bool Classification { get; }
        public List<CrossValidationRow> Rows { get; } = new();
        private readonly List<double> trainActual = new(), trainPredicted = new();
        private readonly List<double> testActual = new(), testPredicted = new();
        public CrossValidationReport(bool classification = false) => Classification = classification;

        public void AddRegression(double[] actualTrain, double[] predictedTrain, double[] actualTest, double[] predictedTest)
        {
            Rows.Add(new CrossValidationRow { Fold = "Fold " + (Rows.Count + 1),
                TrainRows = actualTrain.Length, TestRows = actualTest.Length,
                TrainScore = R2(actualTrain, predictedTrain), TestScore = R2(actualTest, predictedTest) });
            trainActual.AddRange(actualTrain); trainPredicted.AddRange(predictedTrain);
            testActual.AddRange(actualTest); testPredicted.AddRange(predictedTest);
        }

        public void AddClassification(int trainCount, int testCount, double trainAccuracy, double testAccuracy)
        {
            Rows.Add(new CrossValidationRow { Fold = "Fold " + (Rows.Count + 1),
                TrainRows = trainCount, TestRows = testCount, TrainScore = trainAccuracy, TestScore = testAccuracy });
        }

        public void Complete()
        {
            if (Rows.Count == 0 || Rows.Last().Fold == "Combined") return;
            int tr = Rows.Sum(r => r.TrainRows), te = Rows.Sum(r => r.TestRows);
            Rows.Add(new CrossValidationRow { Fold = "Combined", TrainRows = tr, TestRows = te,
                TrainScore = Classification ? Rows.Sum(r => r.TrainScore * r.TrainRows) / tr : R2(trainActual, trainPredicted),
                TestScore = Classification ? Rows.Sum(r => r.TestScore * r.TestRows) / te : R2(testActual, testPredicted) });
            trainActual.Clear(); trainPredicted.Clear(); testActual.Clear(); testPredicted.Clear();
        }

        private static double R2(IReadOnlyList<double> actual, IReadOnlyList<double> predicted)
        {
            if (actual.Count < 2) return double.NaN;
            double mean = actual.Average(), sst = 0, sse = 0;
            for (int i = 0; i < actual.Count; i++)
            {
                sst += Math.Pow(actual[i] - mean, 2);
                sse += Math.Pow(actual[i] - predicted[i], 2);
            }
            return sst == 0 ? double.NaN : 1 - sse / sst;
        }
    }
}
