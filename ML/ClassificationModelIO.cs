using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ArchiAI.ML
{
    // Serializable decision-tree node (for saving Random Forest classifiers).
    public class ClsTreeDTO
    {
        public bool IsLeaf { get; set; }
        public int ClassIdx { get; set; }
        public int Feature { get; set; }
        public double Threshold { get; set; }
        public ClsTreeDTO Left { get; set; }
        public ClsTreeDTO Right { get; set; }
    }

    // Everything needed to reconstruct a classifier's prediction function.
    public class ClassifierState
    {
        public int Type { get; set; }
        public double[] Classes { get; set; }
        public double[] Means { get; set; }
        public double[] Stds { get; set; }
        public int NFeat { get; set; }
        public int NClasses { get; set; }
        public int K { get; set; }
        public double[][] LogW { get; set; }        // logistic (one-vs-rest)
        public double[][] SvmW { get; set; }        // linear SVM (one-vs-rest)
        public double[][] NbMean { get; set; }      // naive bayes
        public double[][] NbVar { get; set; }
        public double[] NbPrior { get; set; }
        public double[][] KnnX { get; set; }
        public int[] KnnY { get; set; }
        public ClsTreeDTO[] Forest { get; set; }    // random forest
        public double[][][] AnnW { get; set; }      // neural net
        public double[][] AnnB { get; set; }
        public int AnnTotal { get; set; }
        public string MlnetB64 { get; set; }        // FastTree / LightGBM
        public bool Lightgbm { get; set; }
        public string XgbB64 { get; set; }          // XGBoost
    }

    // ML.NET prediction I/O (shared by engine + rebuild).
    public class McInput { public float[] Features { get; set; } public float Label { get; set; } }
    public class McOutput { [ColumnName("PredLabel")] public float PredLabel { get; set; } }

    public static class ClassifierRebuild
    {
        private static int Clamp(int r, int n) => r < 0 ? 0 : (r >= n ? n - 1 : r);

        public static Func<double[], int> Build(ClassifierState st)
        {
            int nFeat = st.NFeat, nClasses = st.NClasses;
            switch ((ClassifierType)st.Type)
            {
                case ClassifierType.LogisticRegression:
                    return x => { double best = double.NegativeInfinity; int bi = 0; for (int c = 0; c < nClasses; c++) { var w = st.LogW[c]; double z = w[nFeat]; for (int j = 0; j < nFeat; j++) z += w[j] * x[j]; double p = 1.0 / (1.0 + Math.Exp(-z)); if (p > best) { best = p; bi = c; } } return bi; };
                case ClassifierType.KNN:
                    return x => { var d = new (double dist, int cls)[st.KnnX.Length]; for (int i = 0; i < st.KnnX.Length; i++) { double s = 0; for (int j = 0; j < nFeat; j++) { double e = x[j] - st.KnnX[i][j]; s += e * e; } d[i] = (s, st.KnnY[i]); } return d.OrderBy(t => t.dist).Take(Math.Max(1, st.K)).GroupBy(t => t.cls).OrderByDescending(g => g.Count()).ThenBy(g => g.Key).First().Key; };
                case ClassifierType.NaiveBayes:
                    return x => { double best = double.NegativeInfinity; int bi = 0; for (int c = 0; c < nClasses; c++) { double lp = Math.Log(st.NbPrior[c] + 1e-12); for (int j = 0; j < nFeat; j++) { double v = st.NbVar[c][j], dd = x[j] - st.NbMean[c][j]; lp += -0.5 * Math.Log(2 * Math.PI * v) - (dd * dd) / (2 * v); } if (lp > best) { best = lp; bi = c; } } return bi; };
                case ClassifierType.SVM:
                    return x => { double best = double.NegativeInfinity; int bi = 0; for (int c = 0; c < nClasses; c++) { var w = st.SvmW[c]; double z = w[nFeat]; for (int j = 0; j < nFeat; j++) z += w[j] * x[j]; if (z > best) { best = z; bi = c; } } return bi; };
                case ClassifierType.RandomForest:
                    return x => { var votes = new int[nClasses]; foreach (var t in st.Forest) votes[PredDTO(t, x)]++; int bi = 0; for (int c = 1; c < nClasses; c++) if (votes[c] > votes[bi]) bi = c; return bi; };
                case ClassifierType.ANN:
                    return x => Clamp(AnnForward(st, x), nClasses);
                case ClassifierType.FastTree:
                case ClassifierType.LightGBM:
                    return BuildMlNet(st);
                default: // XGBoost
                    return BuildXgb(st);
            }
        }

        private static int PredDTO(ClsTreeDTO n, double[] x) { while (!n.IsLeaf) n = x[n.Feature] < n.Threshold ? n.Left : n.Right; return n.ClassIdx; }

        private static int AnnForward(ClassifierState st, double[] x)
        {
            double Relu(double z) => z > 0 ? z : 0;
            double[] a = x;
            for (int l = 0; l < st.AnnTotal; l++)
            {
                int inS = st.AnnW[l].Length, outS = st.AnnW[l][0].Length;
                var na = new double[outS];
                for (int j = 0; j < outS; j++) { double s = st.AnnB[l][j]; for (int k = 0; k < inS; k++) s += a[k] * st.AnnW[l][k][j]; na[j] = s; }
                if (l == st.AnnTotal - 1) { double mx = na.Max(), sum = 0; for (int j = 0; j < outS; j++) { na[j] = Math.Exp(na[j] - mx); sum += na[j]; } for (int j = 0; j < outS; j++) na[j] /= sum; }
                else for (int j = 0; j < outS; j++) na[j] = Relu(na[j]);
                a = na;
            }
            int bi = 0; for (int j = 1; j < a.Length; j++) if (a[j] > a[bi]) bi = j; return bi;
        }

        private static Func<double[], int> BuildMlNet(ClassifierState st)
        {
            var ml = new MLContext(seed: 42);
            using var ms = new MemoryStream(Convert.FromBase64String(st.MlnetB64));
            var model = ml.Model.Load(ms, out _);
            var sd = SchemaDefinition.Create(typeof(McInput));
            sd["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, st.NFeat);
            var pe = ml.Model.CreatePredictionEngine<McInput, McOutput>(model, inputSchemaDefinition: sd);
            int nc = st.NClasses;
            return x => { var o = pe.Predict(new McInput { Features = x.Select(v => (float)v).ToArray(), Label = 0 }); return Clamp((int)Math.Round(o.PredLabel), nc); };
        }

        private static Func<double[], int> BuildXgb(ClassifierState st)
        {
            var tmp = Path.GetTempFileName();
            XGBoostSharp.XGBClassifier clf = null;
            try { File.WriteAllBytes(tmp, Convert.FromBase64String(st.XgbB64)); clf = XGBoostSharp.XGBClassifier.LoadFromFile(tmp); }
            finally { try { File.Delete(tmp); } catch { } }
            int nc = st.NClasses;
            return x =>
            {
                // multi:softmax -> PredictRaw returns the class index (raw Predict thresholds at 0.5, binary-only)
                var p = clf.PredictRaw(new float[][] { x.Select(v => (float)v).ToArray() });
                return Clamp((int)Math.Round(p[0]), nc);
            };
        }
    }

    public static class ClassificationIO
    {
        private static readonly JsonSerializerOptions J = new JsonSerializerOptions { IncludeFields = true };
        private class Bundle { public ClassifierState State { get; set; } public string[] Features { get; set; } public string Target { get; set; } public PredictionRecipe Pipeline { get; set; } }

        public static void Save(ClassificationResult r, string[] featureNames, string target, string path)
        {
            if (r?.Model == null) throw new Exception("This model can't be saved (no state).");
            File.WriteAllText(path, JsonSerializer.Serialize(new Bundle { State = r.Model, Features = featureNames, Target = r.TargetName ?? target, Pipeline = r.Pipeline }, J));
        }

        public static ClassificationResult Load(string path, out string[] featureNames, out string target)
        {
            var b = JsonSerializer.Deserialize<Bundle>(File.ReadAllText(path), J);
            featureNames = b.Features ?? new string[0];
            target = b.Target;
            var st = b.State;
            var res = new ClassificationResult
            {
                Pipeline = b.Pipeline,
                TargetName = b.Target,
                Type = (ClassifierType)st.Type,
                Classes = st.Classes,
                FeatureMeans = st.Means,
                FeatureStds = st.Stds,
                FeatureNames = featureNames,
                Model = st,
                ClassifyStd = ClassifierRebuild.Build(st),
                ConvergenceMessage = "loaded model"
            };
            return res;
        }
    }
}
