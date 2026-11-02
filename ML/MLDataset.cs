using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace ArchiAI.ML
{
    public class MLDataset
    {
        public string[] Headers { get; private set; }
        public double[][] Rows { get; private set; }
        public int RowCount => Rows.Length;
        public int ColCount => Headers.Length;

        // Parsed raw strings for preview
        public string[][] RawRows { get; private set; }

        public static MLDataset LoadCsv(string path)
        {
            var lines = File.ReadAllLines(path).Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();
            if (lines.Length < 2) throw new Exception("CSV must have header + at least 1 data row.");

            var ds = new MLDataset();
            ds.Headers = lines[0].Split(',').Select(h => h.Trim().Trim('"')).ToArray();

            var rows = new List<double[]>();
            var rawRows = new List<string[]>();

            for (int i = 1; i < lines.Length; i++)
            {
                var parts = lines[i].Split(',');
                if (parts.Length != ds.Headers.Length) continue;

                rawRows.Add(parts.Select(p => p.Trim()).ToArray());

                var nums = new double[parts.Length];
                bool ok = true;
                for (int j = 0; j < parts.Length; j++)
                {
                    if (!double.TryParse(parts[j].Trim(), NumberStyles.Any, CultureInfo.InvariantCulture, out nums[j]))
                    { ok = false; break; }
                }
                if (ok) rows.Add(nums);
            }

            ds.Rows = rows.ToArray();
            ds.RawRows = rawRows.ToArray();
            return ds;
        }

        public double[] GetColumn(int col)
        {
            return Rows.Select(r => r[col]).ToArray();
        }

        // ============ STATISTICS ============
        public double Mean(int col) => GetColumn(col).Average();

        public double Median(int col)
        {
            var sorted = GetColumn(col).OrderBy(x => x).ToArray();
            int n = sorted.Length;
            return n % 2 == 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 : sorted[n / 2];
        }

        public double Mode(int col)
        {
            return GetColumn(col)
                .GroupBy(x => Math.Round(x, 4))
                .OrderByDescending(g => g.Count())
                .First().Key;
        }

        public double StdDev(int col)
        {
            var data = GetColumn(col);
            double mean = data.Average();
            return Math.Sqrt(data.Select(x => (x - mean) * (x - mean)).Average());
        }

        public double Min(int col) => GetColumn(col).Min();
        public double Max(int col) => GetColumn(col).Max();
        public double Range(int col) => Max(col) - Min(col);

        public double Q1(int col)
        {
            var sorted = GetColumn(col).OrderBy(x => x).ToArray();
            return Percentile(sorted, 25);
        }

        public double Q3(int col)
        {
            var sorted = GetColumn(col).OrderBy(x => x).ToArray();
            return Percentile(sorted, 75);
        }

        public double IQR(int col) => Q3(col) - Q1(col);

        public double Skewness(int col)
        {
            var data = GetColumn(col);
            double mean = data.Average();
            double std = StdDev(col);
            if (std == 0) return 0;
            int n = data.Length;
            return data.Select(x => Math.Pow((x - mean) / std, 3)).Sum() * n / ((n - 1.0) * (n - 2.0));
        }

        public double Kurtosis(int col)
        {
            var data = GetColumn(col);
            double mean = data.Average();
            double std = StdDev(col);
            if (std == 0) return 0;
            return data.Select(x => Math.Pow((x - mean) / std, 4)).Average() - 3;
        }

        public int MissingCount(int col)
        {
            // Count from raw data
            if (RawRows == null) return 0;
            return RawRows.Count(r => string.IsNullOrWhiteSpace(r[col]) ||
                !double.TryParse(r[col], NumberStyles.Any, CultureInfo.InvariantCulture, out _));
        }

        // ============ CORRELATION MATRIX ============
        public double[,] CorrelationMatrix()
        {
            int n = ColCount;
            var mat = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    mat[i, j] = Correlation(i, j);
                }
            }
            return mat;
        }

        public double Correlation(int colA, int colB)
        {
            var a = GetColumn(colA);
            var b = GetColumn(colB);
            double meanA = a.Average(), meanB = b.Average();
            double cov = 0, varA = 0, varB = 0;
            for (int i = 0; i < a.Length; i++)
            {
                double da = a[i] - meanA, db = b[i] - meanB;
                cov += da * db;
                varA += da * da;
                varB += db * db;
            }
            double denom = Math.Sqrt(varA * varB);
            return denom == 0 ? 0 : cov / denom;
        }

        // ============ HISTOGRAM ============
        public (double[] edges, int[] counts) Histogram(int col, int bins = 20)
        {
            var data = GetColumn(col);
            double min = data.Min(), max = data.Max();
            if (Math.Abs(max - min) < 1e-10) { max = min + 1; }
            double step = (max - min) / bins;
            var edges = new double[bins + 1];
            var counts = new int[bins];
            for (int i = 0; i <= bins; i++) edges[i] = min + i * step;
            foreach (var val in data)
            {
                int idx = (int)((val - min) / step);
                if (idx >= bins) idx = bins - 1;
                if (idx < 0) idx = 0;
                counts[idx]++;
            }
            return (edges, counts);
        }

        private static double Percentile(double[] sorted, double p)
        {
            double idx = (p / 100.0) * (sorted.Length - 1);
            int lo = (int)Math.Floor(idx);
            int hi = (int)Math.Ceiling(idx);
            if (lo == hi) return sorted[lo];
            return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
        }
    }
}