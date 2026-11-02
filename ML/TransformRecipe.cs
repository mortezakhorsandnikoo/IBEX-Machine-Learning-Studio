using System;
using System.Collections.Generic;
using System.Linq;

namespace ArchiAI.ML
{
    // Records reversible per-column transforms (currently: natural-log for skewed columns)
    // keyed by ORIGINAL column name. The optimization engine works in transformed space;
    // the View calls Inverse(...) so every reported value returns to ORIGINAL units.
    //
    // Log handles non-positive data with a shift: t = ln(x + shift), inverse = exp(t) - shift,
    // where shift = (min <= 0) ? (1 - min) : 0.
    public class TransformRecipe
    {
        public enum Kind { None = 0, Log = 1 }

        private class Col { public Kind Kind; public double Shift; public bool IsInteger; }
        private readonly Dictionary<string, Col> _cols = new Dictionary<string, Col>(StringComparer.OrdinalIgnoreCase);

        public bool Any => _cols.Values.Any(c => c.Kind != Kind.None);
        public IEnumerable<string> LoggedColumns => _cols.Where(kv => kv.Value.Kind == Kind.Log).Select(kv => kv.Key);

        // Register a log transform for a column, computing the shift from its raw values.
        public void SetLog(string column, IEnumerable<double> rawValues, bool isInteger)
        {
            double min = rawValues.DefaultIfEmpty(0).Min();
            double shift = min <= 0 ? (1 - min) : 0;
            _cols[column] = new Col { Kind = Kind.Log, Shift = shift, IsInteger = isInteger };
        }

        public bool IsLogged(string column) => _cols.TryGetValue(column, out var c) && c.Kind == Kind.Log;

        // raw -> transformed (used when preparing the training/optimization data)
        public double Forward(string column, double raw)
        {
            if (!_cols.TryGetValue(column, out var c) || c.Kind != Kind.Log) return raw;
            double v = raw + c.Shift;
            return Math.Log(v <= 1e-12 ? 1e-12 : v);
        }

        // transformed -> raw ORIGINAL units (used for every reported value / plot).
        public double Inverse(string column, double transformed)
        {
            if (!_cols.TryGetValue(column, out var c) || c.Kind != Kind.Log) return transformed;
            double raw = Math.Exp(transformed) - c.Shift;
            if (c.IsInteger) raw = Math.Round(raw);
            return raw;
        }

        // Sample skewness (Fisher-Pearson) of a numeric column.
        public static double Skewness(IReadOnlyList<double> x)
        {
            int n = x.Count;
            if (n < 3) return 0;
            double mean = x.Average();
            double s2 = 0; for (int i = 0; i < n; i++) { double d = x[i] - mean; s2 += d * d; }
            double sd = Math.Sqrt(s2 / n);
            if (sd < 1e-12) return 0;
            double s3 = 0; for (int i = 0; i < n; i++) { double d = (x[i] - mean) / sd; s3 += d * d * d; }
            return s3 / n;
        }

        // A column is a good log candidate when it is strongly right-skewed and all-positive-ish.
        public static bool IsSkewed(IReadOnlyList<double> x, double threshold = 1.0)
            => Math.Abs(Skewness(x)) >= threshold;
    }
}
