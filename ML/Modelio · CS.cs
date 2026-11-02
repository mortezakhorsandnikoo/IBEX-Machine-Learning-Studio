using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace ArchiAI.ML
{
    // A trained model reduced to what's needed to score new inputs.
    public class SavedModel
    {
        public PredictionRecipe Pipeline;
        public string[] InputNames => Pipeline?.InputNames ?? FeatureNames;
        public string Type;            // "Linear" or "Polynomial"
        public string TargetName;
        public string[] FeatureNames;  // base features (columns excl. target), in order
        public int Degree = 1;         // 1 for linear
        public double Bias;
        public double[] Weights;       // linear: per feature; poly: per expanded feature
        public double[] Means;         // linear + engine-normalized (else null)
        public double[] Stds;

        // Preprocessing normalization (null if the data wasn't normalized).
        public NormalizationRecipe Norm;

        // Predict from RAW real-world feature values: scale features -> predict -> un-scale target.
        public double PredictRaw(double[] raw)
        {
            if (Pipeline != null) return Pipeline.RestoreTarget(Predict(Pipeline.Transform(raw)));
            double[] x = raw;
            if (Norm != null && FeatureNames != null)
            {
                x = new double[raw.Length];
                for (int i = 0; i < raw.Length; i++)
                    x[i] = Norm.Transform(FeatureNames[i], raw[i]);
            }
            double y = Predict(x);
            if (Norm != null) y = Norm.Inverse(TargetName, y);
            return y;
        }

        public double Predict(double[] x)
        {
            if (string.Equals(Type, "Polynomial", StringComparison.OrdinalIgnoreCase))
            {
                var exp = new List<double>();
                foreach (var v in x)
                {
                    double pow = 1.0;
                    for (int p = 1; p <= Degree; p++) { pow *= v; exp.Add(pow); }
                }
                double yp = Bias;
                for (int i = 0; i < Weights.Length && i < exp.Count; i++) yp += Weights[i] * exp[i];
                return yp;
            }

            double[] xin = x;
            if (Means != null && Stds != null)
            {
                xin = new double[x.Length];
                for (int j = 0; j < x.Length; j++)
                {
                    double s = Stds[j] == 0 ? 1e-10 : Stds[j];
                    xin[j] = (x[j] - Means[j]) / s;
                }
            }
            double y = Bias;
            for (int j = 0; j < Weights.Length && j < xin.Length; j++) y += Weights[j] * xin[j];
            return y;
        }
    }

    // Plain-text persistence (round-trippable, no external libraries).
    public static class ModelIO
    {
        private static readonly CultureInfo INV = CultureInfo.InvariantCulture;

        public static void Save(SavedModel m, string path)
        {
            var sb = new StringBuilder();
            sb.AppendLine("ARCHIAI_MODEL v1");
            sb.AppendLine("Type=" + m.Type);
            if (m.Pipeline != null) sb.AppendLine("Pipeline=" + System.Text.Json.JsonSerializer.Serialize(m.Pipeline));
            sb.AppendLine("Target=" + m.TargetName);
            sb.AppendLine("Degree=" + m.Degree.ToString(INV));
            sb.AppendLine("Features=" + string.Join(",", m.FeatureNames ?? new string[0]));
            sb.AppendLine("Bias=" + m.Bias.ToString("R", INV));
            sb.AppendLine("Weights=" + Join(m.Weights));
            sb.AppendLine("Means=" + (m.Means == null ? "NONE" : Join(m.Means)));
            sb.AppendLine("Stds=" + (m.Stds == null ? "NONE" : Join(m.Stds)));
            // Normalization recipe (affine per column)
            if (m.Norm != null)
            {
                sb.AppendLine("NormCols=" + string.Join(",", m.Norm.Columns ?? new string[0]));
                sb.AppendLine("NormA=" + Join(m.Norm.A));
                sb.AppendLine("NormB=" + Join(m.Norm.B));
            }
            File.WriteAllText(path, sb.ToString());
        }

        public static SavedModel Load(string path)
        {
            var map = new Dictionary<string, string>();
            foreach (var line in File.ReadAllLines(path))
            {
                int eq = line.IndexOf('=');
                if (eq < 0) continue;
                map[line.Substring(0, eq).Trim()] = line.Substring(eq + 1).Trim();
            }

            var m = new SavedModel
            {
                Type = Get(map, "Type", "Linear"),
                TargetName = Get(map, "Target", "Target"),
                Degree = int.TryParse(Get(map, "Degree", "1"), NumberStyles.Any, INV, out int d) ? d : 1,
                FeatureNames = Get(map, "Features", "").Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries),
                Bias = ParseD(Get(map, "Bias", "0")),
                Weights = ParseArr(Get(map, "Weights", "")),
                Means = ParseNullableArr(Get(map, "Means", "NONE")),
                Stds = ParseNullableArr(Get(map, "Stds", "NONE"))
            };

            if (map.ContainsKey("NormCols"))
            {
                m.Norm = new NormalizationRecipe
                {
                    Columns = Get(map, "NormCols", "").Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries),
                    A = ParseArr(Get(map, "NormA", "")),
                    B = ParseArr(Get(map, "NormB", ""))
                };
            }
            if (map.TryGetValue("Pipeline", out var pipeline))
                m.Pipeline = System.Text.Json.JsonSerializer.Deserialize<PredictionRecipe>(pipeline);
            return m;
        }

        private static string Join(double[] a) =>
            a == null ? "" : string.Join(",", a.Select(v => v.ToString("R", INV)));

        private static double ParseD(string s) =>
            double.TryParse(s, NumberStyles.Any, INV, out double v) ? v : 0;

        private static double[] ParseArr(string s) =>
            s.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries).Select(ParseD).ToArray();

        private static double[] ParseNullableArr(string s) =>
            (string.IsNullOrWhiteSpace(s) || s == "NONE") ? null : ParseArr(s);

        private static string Get(Dictionary<string, string> m, string k, string dflt) =>
            m.TryGetValue(k, out string v) ? v : dflt;
    }
}
