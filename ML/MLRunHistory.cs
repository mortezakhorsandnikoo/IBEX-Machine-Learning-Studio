using System;
using System.Collections.Generic;

namespace ArchiAI.ML
{
    // One trained-model run, recorded for the Compare tab.
    public class MLRun
    {
        public string Task { get; set; }        // "Regression" | "Classification"
        public string Algorithm { get; set; }
        public string When { get; set; }        // timestamp
        public string Target { get; set; }
        // Regression metrics (NaN if not applicable)
        public double R2 { get; set; } = double.NaN;
        public double AdjR2 { get; set; } = double.NaN;
        public double MAE { get; set; } = double.NaN;
        public double RMSE { get; set; } = double.NaN;
        // Classification metrics (NaN if not applicable)
        public double Accuracy { get; set; } = double.NaN;
        public double MacroF1 { get; set; } = double.NaN;
    }

    // Global, in-memory history of model runs (persists while the app is open).
    public static class MLRunHistory
    {
        public static readonly List<MLRun> Runs = new List<MLRun>();

        public static void AddRegression(string algo, string target, double r2, double adjR2, double mae, double rmse)
        {
            Runs.Add(new MLRun
            {
                Task = "Regression", Algorithm = algo, Target = target,
                When = DateTime.Now.ToString("HH:mm:ss"),
                R2 = r2, AdjR2 = adjR2, MAE = mae, RMSE = rmse
            });
        }

        public static void AddClassification(string algo, string target, double accuracy, double macroF1)
        {
            Runs.Add(new MLRun
            {
                Task = "Classification", Algorithm = algo, Target = target,
                When = DateTime.Now.ToString("HH:mm:ss"),
                Accuracy = accuracy, MacroF1 = macroF1
            });
        }

        public static void Clear() => Runs.Clear();
    }
}
