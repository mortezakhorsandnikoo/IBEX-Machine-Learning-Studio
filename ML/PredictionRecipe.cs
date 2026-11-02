using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace ArchiAI.ML
{
    public sealed class PreprocessingStep
    {
        public NormalizationRecipe Normalization { get; set; }
        public string[] Inputs { get; set; }
        public string[] Outputs { get; set; }
        public double[] Means { get; set; }
        public double[] Scales { get; set; }
        public double[][] Loadings { get; set; }

        public static PreprocessingStep FromPca(PcaResult result) => new PreprocessingStep
        {
            Inputs = result.OriginalHeaders,
            Outputs = Enumerable.Range(1, result.ComponentsKept).Select(i => "PC" + i).ToArray(),
            Means = result.FeatureMeans, Scales = result.FeatureScales,
            Loadings = result.Eigenvectors.Select(row => row.Take(result.ComponentsKept).ToArray()).ToArray()
        };
    }

    // A model owns an independent snapshot of the transforms used for its training data.
    public sealed class PredictionRecipe
    {
        public string[] InputNames { get; set; }
        public string[] ModelFeatures { get; set; }
        public string TargetName { get; set; }
        public List<PreprocessingStep> Steps { get; set; } = new();

        public static PredictionRecipe Create(string[] features, string target, IEnumerable<PreprocessingStep> steps)
        {
            var copy = JsonSerializer.Deserialize<List<PreprocessingStep>>(JsonSerializer.Serialize(steps));
            var needed = new HashSet<string>(features);
            foreach (var step in copy.AsEnumerable().Reverse())
            {
                if (step.Normalization != null) continue;
                if (target != null && step.Outputs.Contains(target))
                    throw new InvalidOperationException("Preserve the target column when applying PCA before supervised training. Re-import the original data and enable ‘Last column is target (Y)’.");
                if (!step.Outputs.Any(needed.Contains)) continue;
                needed.ExceptWith(step.Outputs); needed.UnionWith(step.Inputs);
            }
            if (target != null && needed.Contains(target))
                throw new InvalidOperationException("The target was included in PCA. Reapply PCA with the target column preserved.");
            var order = copy.SelectMany(s => s.Normalization?.Columns ?? s.Inputs).Concat(features).Distinct();
            return new PredictionRecipe { InputNames = order.Where(needed.Contains).ToArray(),
                ModelFeatures = features.ToArray(), TargetName = target, Steps = copy };
        }

        public double[] Transform(double[] raw)
        {
            if (raw.Length != InputNames.Length || raw.Any(v => !double.IsFinite(v)))
                throw new ArgumentException("Enter a finite number for every original feature.");
            var values = InputNames.Select((name, i) => (name, value: raw[i])).ToDictionary(p => p.name, p => p.value);
            foreach (var step in Steps)
            {
                if (step.Normalization != null)
                {
                    foreach (string name in values.Keys.ToArray()) values[name] = step.Normalization.Transform(name, values[name]);
                    continue;
                }
                var projected = new double[step.Outputs.Length];
                for (int c = 0; c < projected.Length; c++)
                    for (int f = 0; f < step.Inputs.Length; f++)
                        projected[c] += (values[step.Inputs[f]] - step.Means[f]) / step.Scales[f] * step.Loadings[f][c];
                foreach (var name in step.Inputs) values.Remove(name);
                for (int c = 0; c < projected.Length; c++) values[step.Outputs[c]] = projected[c];
            }
            return ModelFeatures.Select(name => values[name]).ToArray();
        }

        public double RestoreTarget(double prediction)
        {
            foreach (var step in Steps.AsEnumerable().Reverse())
                if (TargetName != null && step.Normalization != null)
                    prediction = step.Normalization.Inverse(TargetName, prediction);
            return prediction;
        }
    }
}
