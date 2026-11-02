using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.Statistics;

namespace ArchiAI.ML
{
    // Captures the per-column normalization as an affine map:  normalized = A*raw + B.
    // Z-Score, Min-Max and Max-Abs are all affine, so A and B are recovered by comparing
    // original vs normalized values — this matches whatever PreprocessingEngine did exactly.
    public class NormalizationRecipe
    {
        public static NormalizationRecipe ForDataset(MLDataset dataset, NormMethod method)
        {
            var a = new double[dataset.ColCount]; var b = new double[dataset.ColCount];
            for (int c = 0; c < a.Length; c++)
            {
                var values = dataset.Rows.Select(row => row[c]).ToArray();
                double offset = 0, scale = 1;
                switch (method)
                {
                    case NormMethod.ZScore: offset = values.Average(); scale = values.StandardDeviation(); break;
                    case NormMethod.MinMax: offset = values.Min(); scale = values.Max() - offset; break;
                    case NormMethod.MaxAbs: scale = values.Select(Math.Abs).Max(); break;
                }
                if (scale == 0) scale = 1e-8;
                a[c] = 1 / scale; b[c] = -offset / scale;
            }
            return new NormalizationRecipe { Columns = dataset.Headers.ToArray(), A = a, B = b };
        }
        public string[] Columns { get; set; }
        public double[] A { get; set; }
        public double[] B { get; set; }

        private Dictionary<string, int> _idx;

        private int Index(string col)
        {
            if (_idx == null)
            {
                _idx = new Dictionary<string, int>();
                for (int i = 0; i < (Columns?.Length ?? 0); i++) _idx[Columns[i]] = i;
            }
            return _idx.TryGetValue(col, out int k) ? k : -1;
        }

        // raw -> normalized
        public double Transform(string col, double raw)
        {
            int i = Index(col);
            return i < 0 ? raw : A[i] * raw + B[i];
        }

        // normalized -> raw
        public double Inverse(string col, double norm)
        {
            int i = Index(col);
            if (i < 0) return norm;
            return A[i] == 0 ? norm : (norm - B[i]) / A[i];
        }

        // Build by fitting A,B per column from original vs normalized rows (same order).
        public static NormalizationRecipe Fit(string[] headers, double[][] original, double[][] normalized)
        {
            int cols = headers.Length;
            var a = new double[cols];
            var b = new double[cols];
            int rows = Math.Min(original.Length, normalized.Length);

            for (int c = 0; c < cols; c++)
            {
                a[c] = 1; b[c] = 0; // identity fallback (constant column)
                for (int i = 0; i < rows; i++)
                {
                    for (int j = i + 1; j < rows; j++)
                    {
                        double dx = original[i][c] - original[j][c];
                        if (Math.Abs(dx) < 1e-12) continue;
                        double dz = normalized[i][c] - normalized[j][c];
                        a[c] = dz / dx;
                        b[c] = normalized[i][c] - a[c] * original[i][c];
                        goto done;
                    }
                }
            done:;
            }
            return new NormalizationRecipe { Columns = headers, A = a, B = b };
        }
    }
}
