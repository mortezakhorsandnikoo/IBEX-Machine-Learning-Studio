using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace ArchiAI.ML
{
    // ============ Data container (matches the Dendrite Dataset) ============
    public class Dataset
    {
        public float[][] X { get; set; }
        public float[] y { get; set; }
        public int RowCount => X?.Length ?? 0;
        public int FeatureCount => (X != null && X.Length > 0) ? X[0].Length : 0;
    }

    // ============ Unified model interface ============
    public interface IRegressionModel : IDisposable
    {
        void Train(Dataset data);
        float[] Predict(float[][] X);
        string Name { get; }
        void SetHyperparameters(Dictionary<string, double> parameters);
        Dictionary<string, double> GetDefaultHyperparameters();
        // Full persistence: SaveState returns a self-contained string (JSON or base64 model bytes);
        // LoadState restores a model that can predict without retraining.
        string SaveState();
        void LoadState(string state);
    }

    // Portable single-file model bundle (.aimodel): model type + feature names + state.
    public static class RegModelIO
    {
        public static readonly JsonSerializerOptions J = new JsonSerializerOptions { IncludeFields = true };

        private class Bundle { public string Type { get; set; } public string[] Features { get; set; } public string State { get; set; } public PredictionRecipe Pipeline { get; set; } }

        public static void Save(IRegressionModel model, RegModelType type, string[] featureNames, string path, PredictionRecipe pipeline = null)
        {
            var b = new Bundle { Type = type.ToString(), Features = featureNames ?? new string[0], State = model.SaveState(), Pipeline = pipeline };
            File.WriteAllText(path, JsonSerializer.Serialize(b, J));
        }

        public static IRegressionModel Load(string path, out RegModelType type, out string[] featureNames)
            => Load(path, out type, out featureNames, out _);

        public static IRegressionModel Load(string path, out RegModelType type, out string[] featureNames, out PredictionRecipe pipeline)
        {
            var b = JsonSerializer.Deserialize<Bundle>(File.ReadAllText(path), J);
            pipeline = b.Pipeline;
            type = (RegModelType)Enum.Parse(typeof(RegModelType), b.Type);
            featureNames = b.Features ?? new string[0];
            var m = RegModelFactory.Create(type);
            m.LoadState(b.State);
            return m;
        }
    }

    // ============ Which algorithms are available in the Regression dropdown ============
    // (Linear + Polynomial keep their own dedicated panels; these are the IRegressionModel ones.)
    public enum RegModelType
    {
        NormalEquation,
        LWR,
        RandomForest,
        MLP,
        RBF,
        FastTree,   // ML.NET  (batch 2)
        LightGBM,   // ML.NET  (batch 2)
        OGD,        // ML.NET  (batch 2)
        XGBoost     // XGBoostSharp (batch 2)
    }

    // one hyperparameter spec (for building sliders + validation curve)
    public class HyperSpec
    {
        public string Name;
        public double Lo, Hi, Default;
        public bool IsInteger;
        public HyperSpec(string n, double lo, double hi, double def, bool isInt)
        { Name = n; Lo = lo; Hi = hi; Default = def; IsInteger = isInt; }
    }

    public static class RegModelInfo
    {
        public static string Display(RegModelType t)
        {
            switch (t)
            {
                case RegModelType.NormalEquation: return "Normal Equation";
                case RegModelType.LWR: return "Locally Weighted Regression";
                case RegModelType.RandomForest: return "Random Forest";
                case RegModelType.MLP: return "Neural Net (MLP)";
                case RegModelType.RBF: return "RBF Surrogate";
                case RegModelType.FastTree: return "FastTree (boosted)";
                case RegModelType.LightGBM: return "LightGBM";
                case RegModelType.OGD: return "Online Gradient Descent";
                case RegModelType.XGBoost: return "XGBoost";
                default: return t.ToString();
            }
        }

        public static bool IsTreeFamily(RegModelType t) =>
            t == RegModelType.RandomForest || t == RegModelType.FastTree ||
            t == RegModelType.LightGBM || t == RegModelType.XGBoost;

        // Hyperparameters shown as sliders. First entry = the one used for the Validation Curve.
        public static List<HyperSpec> Hyper(RegModelType t)
        {
            switch (t)
            {
                case RegModelType.NormalEquation:
                    return new List<HyperSpec> { new HyperSpec("ridge", 0, 1, 0.0, false) };
                case RegModelType.LWR:
                    return new List<HyperSpec> {
                        new HyperSpec("tau", 0.1, 5.0, 1.0, false),
                        new HyperSpec("lambda", 0, 0.1, 1e-3, false) };
                case RegModelType.RandomForest:
                    return new List<HyperSpec> {
                        new HyperSpec("trees", 10, 300, 50, true),
                        new HyperSpec("maxDepth", 2, 30, 12, true),
                        new HyperSpec("minLeaf", 1, 50, 10, true) };
                case RegModelType.MLP:
                    return new List<HyperSpec> {
                        new HyperSpec("hiddenSize", 4, 128, 32, true),
                        new HyperSpec("hiddenLayers", 1, 4, 1, true),
                        new HyperSpec("epochs", 50, 1000, 200, true),
                        new HyperSpec("learningRate", 1e-4, 0.1, 1e-3, false) };
                case RegModelType.RBF:
                    return new List<HyperSpec> {
                        new HyperSpec("epsilon", 0.1, 5.0, 1.0, false),
                        new HyperSpec("smoothing", 0.0, 1.0, 0.0, false) };
                case RegModelType.FastTree:
                    return new List<HyperSpec> {
                        new HyperSpec("numberOfTrees", 50, 500, 100, true),
                        new HyperSpec("numberOfLeaves", 10, 50, 31, true),
                        new HyperSpec("learningRate", 0.01, 0.5, 0.2, false),
                        new HyperSpec("minExamplesPerLeaf", 1, 20, 10, true) };
                case RegModelType.LightGBM:
                    return new List<HyperSpec> {
                        new HyperSpec("numberOfIterations", 50, 500, 200, true),
                        new HyperSpec("numberOfLeaves", 10, 50, 31, true),
                        new HyperSpec("learningRate", 0.01, 0.5, 0.1, false),
                        new HyperSpec("minExamplesPerLeaf", 1, 20, 10, true) };
                case RegModelType.OGD:
                    return new List<HyperSpec> {
                        new HyperSpec("iterations", 10, 500, 50, true),
                        new HyperSpec("learningRate", 1e-3, 0.5, 0.01, false) };
                case RegModelType.XGBoost:
                    return new List<HyperSpec> {
                        new HyperSpec("numRounds", 50, 500, 200, true),
                        new HyperSpec("maxDepth", 3, 10, 6, true),
                        new HyperSpec("eta", 0.01, 0.5, 0.1, false),
                        new HyperSpec("subsample", 0.5, 1.0, 0.8, false) };
                default: return new List<HyperSpec>();
            }
        }
    }

    public static class RegModelFactory
    {
        public static IRegressionModel Create(RegModelType t)
        {
            switch (t)
            {
                case RegModelType.NormalEquation: return new NormalEquationModel();
                case RegModelType.LWR: return new LwrModel();
                case RegModelType.RandomForest: return new RandomForestModel();
                case RegModelType.MLP: return new MlpModel();
                case RegModelType.RBF: return new RbfModel();
                case RegModelType.FastTree: return new FastTreeModel();
                case RegModelType.LightGBM: return new LightGBMModel();
                case RegModelType.OGD: return new OgdModel();
                case RegModelType.XGBoost: return new XGBoostModel();
                default: return new NormalEquationModel();
            }
        }
    }

    // ============ Metrics: R², Adjusted R², MAE, RMSE ============
    public struct RegMetrics
    {
        public double R2, AdjR2, MAE, RMSE;
        public override string ToString() =>
            $"R²={R2:F4}  AdjR²={AdjR2:F4}  MAE={MAE:G5}  RMSE={RMSE:G5}";
    }

    public static class RegressionMetrics
    {
        public static RegMetrics Compute(float[] actual, float[] pred, int nFeatures)
        {
            int n = Math.Min(actual.Length, pred.Length);
            if (n == 0) return new RegMetrics();
            double mean = 0; for (int i = 0; i < n; i++) mean += actual[i]; mean /= n;
            double ssTot = 0, ssRes = 0, mae = 0;
            for (int i = 0; i < n; i++)
            {
                double e = actual[i] - pred[i];
                ssRes += e * e;
                mae += Math.Abs(e);
                double d = actual[i] - mean;
                ssTot += d * d;
            }
            double r2 = ssTot <= 1e-12 ? 0 : 1.0 - ssRes / ssTot;
            double adj = r2;
            if (n - nFeatures - 1 > 0)
                adj = 1.0 - (1.0 - r2) * (n - 1.0) / (n - nFeatures - 1.0);
            return new RegMetrics
            {
                R2 = r2,
                AdjR2 = adj,
                MAE = mae / n,
                RMSE = Math.Sqrt(ssRes / n)
            };
        }
    }

    // ============ Build Dataset (float) from the preprocessed MLDataset ============
    public static class DatasetBuilder
    {
        public static void Build(MLDataset ds, int targetCol,
            out float[][] X, out float[] y, out string[] featureNames)
        {
            int rows = ds.RowCount, cols = ds.ColCount, nf = cols - 1;
            X = new float[rows][];
            y = new float[rows];
            var names = new List<string>();
            for (int j = 0; j < cols; j++) if (j != targetCol) names.Add(ds.Headers[j]);
            featureNames = names.ToArray();
            for (int i = 0; i < rows; i++)
            {
                X[i] = new float[nf];
                int f = 0;
                for (int j = 0; j < cols; j++)
                {
                    if (j == targetCol) { y[i] = (float)ds.Rows[i][j]; continue; }
                    X[i][f++] = (float)ds.Rows[i][j];
                }
            }
        }

        public static Dataset Subset(float[][] X, float[] y, int[] idx)
        {
            var nx = new float[idx.Length][];
            var ny = new float[idx.Length];
            for (int i = 0; i < idx.Length; i++) { nx[i] = (float[])X[idx[i]].Clone(); ny[i] = y[idx[i]]; }
            return new Dataset { X = nx, y = ny };
        }

        public static int[] Shuffle(int n, int seed = 42)
        {
            var idx = Enumerable.Range(0, n).ToArray();
            var r = new Random(seed);
            for (int i = n - 1; i > 0; i--) { int j = r.Next(i + 1); (idx[i], idx[j]) = (idx[j], idx[i]); }
            return idx;
        }
    }
}
