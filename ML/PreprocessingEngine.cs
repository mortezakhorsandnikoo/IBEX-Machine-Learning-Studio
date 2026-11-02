using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using static ArchiAI.Views.MLView;

namespace ArchiAI.ML
{
    // ==================== DESCRIBE ====================
    public class DescribeResult
    {
        public string[] Headers { get; set; }
        public int[] Count { get; set; }
        public double[] Mean { get; set; }
        public double[] StdDev { get; set; }
        public double[] Min { get; set; }
        public double[] Q1 { get; set; }
        public double[] Median { get; set; }
        public double[] Q3 { get; set; }
        public double[] Max { get; set; }
        public double[] Skewness { get; set; }
        public double[] Kurtosis { get; set; }
        public int[] MissingCount { get; set; }
    }

    // ==================== GAUSSIAN ANOMALY ====================
    public class AnomalyResult
    {
        public double[] Scores { get; set; }
        public double[] MahalanobisDistances { get; set; }
        public int[] AnomalyIndices { get; set; }
        public double[,] CovarianceMatrix { get; set; }
        public int TotalRows { get; set; }
        public int AnomalyCount { get; set; }
        public double Threshold { get; set; }
        public double[][] FilteredData { get; set; }
        public string[] Headers { get; set; }
    }

    // ==================== MISSING VALUE ====================
    public class MissingValueReport
    {
        public string[] Headers { get; set; }
        public int[] MissingPerColumn { get; set; }
        public int[] MissingRowIndices { get; set; }
        public int TotalMissing { get; set; }
        public int TotalRows { get; set; }
        public double MissingPercent { get; set; }
        public bool HasMissing { get; set; }
    }

    public enum MissingValueMethod { Mean = 0, Max = 1, Min = 2, Mode = 3, RemoveRow = 4 }

    // ==================== NORMALIZE ====================
    public enum NormMethod { ZScore = 0, MinMax = 1, MaxAbs = 2 }

    public class NormalizeResult
    {
        public double[][] NormalizedData { get; set; }
        public string[] Headers { get; set; }
        public string MethodName { get; set; }
    }

    // ==================== PCA ====================
    public class PcaResult
    {
        public double[] FeatureMeans { get; set; }
        public double[] FeatureScales { get; set; }
        public double[][] ProjectedData { get; set; }
        public double[][] ReconstructedData { get; set; }
        public double[][] Eigenvectors { get; set; }
        public double[] Eigenvalues { get; set; }
        public double[][] CovarianceMatrix { get; set; }
        public int ComponentsKept { get; set; }
        public double VarianceRetained { get; set; }
        public double[] ExplainedVarianceRatio { get; set; }
        public double[] CumulativeVariance { get; set; }
        public string[] OriginalHeaders { get; set; }
    }

    // ==================== ENGINE ====================
    public static class PreprocessingEngine
    {
        // ================ DESCRIBE ================
        public static DescribeResult Describe(MLDataset dataset)
        {
            int n = dataset.ColCount;
            var result = new DescribeResult
            {
                Headers = dataset.Headers.ToArray(),
                Count = new int[n],
                Mean = new double[n],
                StdDev = new double[n],
                Min = new double[n],
                Q1 = new double[n],
                Median = new double[n],
                Q3 = new double[n],
                Max = new double[n],
                Skewness = new double[n],
                Kurtosis = new double[n],
                MissingCount = new int[n]
            };

            for (int i = 0; i < n; i++)
            {
                var col = dataset.GetColumn(i);
                result.Count[i] = col.Length;
                result.Mean[i] = col.Average();
                result.StdDev[i] = dataset.StdDev(i);
                result.Min[i] = col.Min();
                result.Max[i] = col.Max();
                result.Q1[i] = dataset.Q1(i);
                result.Median[i] = dataset.Median(i);
                result.Q3[i] = dataset.Q3(i);
                result.Skewness[i] = dataset.Skewness(i);
                result.Kurtosis[i] = dataset.Kurtosis(i);
                result.MissingCount[i] = dataset.MissingCount(i);
            }

            return result;
        }

        // ================ GAUSSIAN ANOMALY DETECTION ================
        public static AnomalyResult DetectAnomalies(MLDataset dataset, double threshold)
        {
            int rows = dataset.RowCount;
            int cols = dataset.ColCount;

            var matrix = Matrix<double>.Build.Dense(rows, cols,
                (i, j) => dataset.Rows[i][j]);

            // Z-Score normalize
            var normalized = ZScoreNormalize(matrix);

            int n = normalized.RowCount;
            int p = normalized.ColumnCount;

            // Mean vector
            var meanVec = Vector<double>.Build.Dense(p,
                j => normalized.Column(j).Average());

            // Covariance matrix
            var ones = Vector<double>.Build.Dense(n, 1.0);
            var centered = normalized - ones.ToColumnMatrix() * meanVec.ToRowMatrix();
            var covMatrix = (centered.TransposeThisAndMultiply(centered)) / (n - 1);

            // Inverse and determinant
            var covInv = covMatrix.Inverse();
            double det = covMatrix.Determinant();
            double normConst = 1.0 / Math.Sqrt(Math.Pow(2 * Math.PI, p) * Math.Abs(det));

            // Mahalanobis distances
            var mdSquared = (centered * covInv * centered.Transpose()).Diagonal();
            var mahalanobis = mdSquared.Map(Math.Sqrt).ToArray();
            var probabilities = mdSquared.Map(md2 => normConst * Math.Exp(-0.5 * md2)).ToArray();

            // Find anomalies
            var anomalyIdx = new List<int>();
            for (int i = 0; i < probabilities.Length; i++)
                if (probabilities[i] < threshold) anomalyIdx.Add(i);

            // Filtered data
            var filtered = new List<double[]>();
            for (int i = 0; i < rows; i++)
                if (!anomalyIdx.Contains(i))
                    filtered.Add(dataset.Rows[i]);

            // Covariance to 2D array
            var cov2d = new double[p, p];
            for (int i = 0; i < p; i++)
                for (int j = 0; j < p; j++)
                    cov2d[i, j] = covMatrix[i, j];

            return new AnomalyResult
            {
                Scores = probabilities,
                MahalanobisDistances = mahalanobis,
                AnomalyIndices = anomalyIdx.ToArray(),
                CovarianceMatrix = cov2d,
                TotalRows = rows,
                AnomalyCount = anomalyIdx.Count,
                Threshold = threshold,
                FilteredData = filtered.ToArray(),
                Headers = dataset.Headers.ToArray()
            };
        }

        // ================ MISSING VALUE DETECTION ================
        public static MissingValueReport DetectMissing(string csvPath)
        {
            var lines = File.ReadAllLines(csvPath)
                .Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();

            if (lines.Length < 2)
                return new MissingValueReport
                {
                    HasMissing = false,
                    TotalRows = 0,
                    TotalMissing = 0
                };

            var headers = lines[0].Split(',').Select(h => h.Trim().Trim('"')).ToArray();
            int colCount = headers.Length;
            var missingPerCol = new int[colCount];
            var missingRows = new List<int>();
            int totalMissing = 0;

            for (int i = 1; i < lines.Length; i++)
            {
                var values = lines[i].Split(',');
                bool rowHasMissing = false;
                for (int j = 0; j < Math.Min(values.Length, colCount); j++)
                {
                    if (string.IsNullOrWhiteSpace(values[j]) ||
                        !double.TryParse(values[j].Trim(), NumberStyles.Any,
                            CultureInfo.InvariantCulture, out _))
                    {
                        missingPerCol[j]++;
                        totalMissing++;
                        rowHasMissing = true;
                    }
                }
                if (rowHasMissing) missingRows.Add(i);
            }

            int dataRows = lines.Length - 1;
            return new MissingValueReport
            {
                Headers = headers,
                MissingPerColumn = missingPerCol,
                MissingRowIndices = missingRows.ToArray(),
                TotalMissing = totalMissing,
                TotalRows = dataRows,
                MissingPercent = dataRows > 0 ? (double)totalMissing / (dataRows * colCount) * 100 : 0,
                HasMissing = totalMissing > 0
            };
        }

        // ================ MISSING VALUE HANDLING ================
        public static (string[][] cleanedData, string[] headers, int removedCount)
            HandleMissing(string csvPath, MissingValueMethod method)
        {
            var lines = File.ReadAllLines(csvPath)
                .Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();

            if (lines.Length < 2) return (new string[0][], new string[0], 0);

            var headers = lines[0].Split(',').Select(h => h.Trim()).ToArray();
            int colCount = headers.Length;

            var data = new List<string[]>();
            for (int i = 1; i < lines.Length; i++)
                data.Add(lines[i].Split(',').Select(v => v.Trim()).ToArray());

            // Compute column stats from valid values
            var colValues = new List<double>[colCount];
            for (int j = 0; j < colCount; j++)
            {
                colValues[j] = new List<double>();
                foreach (var row in data)
                {
                    if (j < row.Length && double.TryParse(row[j],
                        NumberStyles.Any, CultureInfo.InvariantCulture, out double v))
                        colValues[j].Add(v);
                }
            }

            var stats = new double[colCount];
            for (int j = 0; j < colCount; j++)
            {
                if (colValues[j].Count == 0) { stats[j] = 0; continue; }
                switch (method)
                {
                    case MissingValueMethod.Mean: stats[j] = colValues[j].Average(); break;
                    case MissingValueMethod.Max: stats[j] = colValues[j].Max(); break;
                    case MissingValueMethod.Min: stats[j] = colValues[j].Min(); break;
                    case MissingValueMethod.Mode:
                        stats[j] = colValues[j].GroupBy(x => x)
                            .OrderByDescending(g => g.Count()).First().Key;
                        break;
                }
            }

            var cleaned = new List<string[]>();
            int removed = 0;

            foreach (var row in data)
            {
                bool hasMissing = false;
                var newRow = new string[colCount];

                for (int j = 0; j < colCount; j++)
                {
                    string val = j < row.Length ? row[j] : "";
                    if (string.IsNullOrWhiteSpace(val) ||
                        !double.TryParse(val, NumberStyles.Any, CultureInfo.InvariantCulture, out _))
                    {
                        hasMissing = true;
                        if (method == MissingValueMethod.RemoveRow) break;
                        newRow[j] = stats[j].ToString("G10", CultureInfo.InvariantCulture);
                    }
                    else
                    {
                        newRow[j] = val;
                    }
                }

                if (method == MissingValueMethod.RemoveRow && hasMissing)
                {
                    removed++;
                    continue;
                }
                cleaned.Add(newRow);
            }

            return (cleaned.ToArray(), headers, removed);
        }

        // ================ NORMALIZE ================
        public static NormalizeResult NormalizeData(MLDataset dataset, NormMethod method)
        {
            int rows = dataset.RowCount;
            int cols = dataset.ColCount;

            var matrix = Matrix<double>.Build.Dense(rows, cols,
                (i, j) => dataset.Rows[i][j]);

            Matrix<double> normalized;
            string methodName;

            switch (method)
            {
                case NormMethod.ZScore:
                    normalized = NormZScore(matrix);
                    methodName = "Z-Score Standardization";
                    break;
                case NormMethod.MinMax:
                    normalized = NormMinMax(matrix);
                    methodName = "Min-Max Normalization";
                    break;
                case NormMethod.MaxAbs:
                    normalized = NormMaxAbs(matrix);
                    methodName = "Max-Absolute Scaling";
                    break;
                default:
                    normalized = matrix;
                    methodName = "None";
                    break;
            }

            var result = new double[rows][];
            for (int i = 0; i < rows; i++)
                result[i] = normalized.Row(i).ToArray();

            return new NormalizeResult
            {
                NormalizedData = result,
                Headers = dataset.Headers.ToArray(),
                MethodName = methodName
            };
        }

        // ================ LABEL ENCODER ================
        public static LabelEncoderResult LabelEncode(string csvPath, int[] targetColumns = null)
        {
            var lines = File.ReadAllLines(csvPath)
                .Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();

            if (lines.Length < 2) return null;

            var headers = lines[0].Split(',').Select(h => h.Trim().Trim('"')).ToArray();
            int colCount = headers.Length;

            // Parse data
            var data = new List<string[]>();
            for (int i = 1; i < lines.Length; i++)
            {
                var vals = lines[i].Split(',').Select(v => v.Trim().Trim('"')).ToArray();
                if (vals.Length < colCount)
                {
                    var padded = new string[colCount];
                    Array.Copy(vals, padded, vals.Length);
                    for (int j = vals.Length; j < colCount; j++) padded[j] = "";
                    data.Add(padded);
                }
                else
                {
                    data.Add(vals.Take(colCount).ToArray());
                }
            }

            // Detect which columns are categorical (non-numeric)
            var categoricalCols = new List<int>();

            if (targetColumns != null && targetColumns.Length > 0)
            {
                // User specified columns
                categoricalCols.AddRange(targetColumns.Where(c => c >= 0 && c < colCount));
            }
            else
            {
                // Auto-detect: column is categorical if majority of values are non-numeric
                for (int j = 0; j < colCount; j++)
                {
                    int nonNumeric = 0;
                    int total = 0;
                    foreach (var row in data)
                    {
                        if (!string.IsNullOrWhiteSpace(row[j]))
                        {
                            total++;
                            if (!double.TryParse(row[j], NumberStyles.Any,
                                CultureInfo.InvariantCulture, out _))
                                nonNumeric++;
                        }
                    }
                    // If more than 50% non-numeric, treat as categorical
                    if (total > 0 && (double)nonNumeric / total > 0.5)
                        categoricalCols.Add(j);
                }
            }

            // Build mappings for each categorical column
            var mappings = new Dictionary<string, Dictionary<string, int>>();
            int totalEncoded = 0;

            foreach (int col in categoricalCols)
            {
                // Get unique values in order of first appearance
                var uniqueValues = new List<string>();
                foreach (var row in data)
                {
                    string val = row[col];
                    if (!string.IsNullOrWhiteSpace(val) && !uniqueValues.Contains(val))
                        uniqueValues.Add(val);
                }

                // Sort alphabetically for consistent encoding
                uniqueValues.Sort(StringComparer.OrdinalIgnoreCase);

                var colMapping = new Dictionary<string, int>();
                for (int i = 0; i < uniqueValues.Count; i++)
                    colMapping[uniqueValues[i]] = i;

                mappings[headers[col]] = colMapping;
            }

            // Encode data
            var encodedData = new List<string[]>();
            foreach (var row in data)
            {
                var newRow = new string[colCount];
                for (int j = 0; j < colCount; j++)
                {
                    if (categoricalCols.Contains(j))
                    {
                        string val = row[j];
                        if (!string.IsNullOrWhiteSpace(val) &&
                            mappings[headers[j]].ContainsKey(val))
                        {
                            newRow[j] = mappings[headers[j]][val]
                                .ToString(CultureInfo.InvariantCulture);
                            totalEncoded++;
                        }
                        else
                        {
                            newRow[j] = "-1"; // unknown
                            totalEncoded++;
                        }
                    }
                    else
                    {
                        newRow[j] = row[j];
                    }
                }
                encodedData.Add(newRow);
            }

            return new LabelEncoderResult
            {
                Headers = headers,
                EncodedData = encodedData.ToArray(),
                Mappings = mappings,
                EncodedColumnIndices = categoricalCols.ToArray(),
                TotalEncoded = totalEncoded
            };
        }

        private static Matrix<double> NormZScore(Matrix<double> m)
        {
            var n = m.Clone();
            for (int c = 0; c < n.ColumnCount; c++)
            {
                var col = n.Column(c);
                double mean = col.Mean();
                double std = col.StandardDeviation();
                if (std == 0) std = 1e-8;
                for (int r = 0; r < n.RowCount; r++)
                    n[r, c] = (n[r, c] - mean) / std;
            }
            return n;
        }

        private static Matrix<double> NormMinMax(Matrix<double> m)
        {
            var n = m.Clone();
            for (int c = 0; c < n.ColumnCount; c++)
            {
                var col = n.Column(c);
                double min = col.Min();
                double max = col.Max();
                double range = max - min;
                if (range == 0) range = 1e-8;
                for (int r = 0; r < n.RowCount; r++)
                    n[r, c] = (n[r, c] - min) / range;
            }
            return n;
        }

        private static Matrix<double> NormMaxAbs(Matrix<double> m)
        {
            var n = m.Clone();
            for (int c = 0; c < n.ColumnCount; c++)
            {
                var col = n.Column(c);
                double maxAbs = col.Select(Math.Abs).Max();
                if (maxAbs == 0) maxAbs = 1e-8;
                for (int r = 0; r < n.RowCount; r++)
                    n[r, c] = n[r, c] / maxAbs;
            }
            return n;
        }

        private static Matrix<double> ZScoreNormalize(Matrix<double> matrix)
        {
            int rows = matrix.RowCount;
            int cols = matrix.ColumnCount;
            var means = Vector<double>.Build.Dense(cols, j => matrix.Column(j).Average());
            var stds = Vector<double>.Build.Dense(cols, j =>
            {
                double s = matrix.Column(j).StandardDeviation();
                return s > 0 ? s : 1e-8;
            });

            var normalized = Matrix<double>.Build.Dense(rows, cols);
            for (int j = 0; j < cols; j++)
                for (int i = 0; i < rows; i++)
                    normalized[i, j] = (matrix[i, j] - means[j]) / stds[j];
            return normalized;
        }

        // ================ PCA ================
        public static PcaResult RunPCA(MLDataset dataset, int mode, double value,
            bool hasTarget)
        {
            int rows = dataset.RowCount;
            int totalCols = dataset.ColCount;
            int featureCols = hasTarget ? totalCols - 1 : totalCols;

            // Extract features
            double[][] X = new double[rows][];
            double[] y = hasTarget ? new double[rows] : null;
            for (int i = 0; i < rows; i++)
            {
                X[i] = new double[featureCols];
                for (int j = 0; j < featureCols; j++)
                    X[i][j] = dataset.Rows[i][j];
                if (hasTarget) y[i] = dataset.Rows[i][totalCols - 1];
            }

            int m = rows;
            int n = featureCols;

            // Standardize
            double[] mu = new double[n];
            double[] std = new double[n];
            for (int j = 0; j < n; j++)
            {
                var col = X.Select(r => r[j]).ToArray();
                mu[j] = col.Average();
                std[j] = col.StandardDeviation();
                if (std[j] == 0) std[j] = 1e-8;
            }

            double[][] Xnorm = new double[m][];
            for (int i = 0; i < m; i++)
            {
                Xnorm[i] = new double[n];
                for (int j = 0; j < n; j++)
                    Xnorm[i][j] = (X[i][j] - mu[j]) / std[j];
            }

            // Covariance matrix
            double[][] Sigma = new double[n][];
            for (int i = 0; i < n; i++)
            {
                Sigma[i] = new double[n];
                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < m; k++)
                        sum += Xnorm[k][i] * Xnorm[k][j];
                    Sigma[i][j] = sum / m;
                }
            }

            // SVD (eigendecomposition of symmetric matrix)
            var mat = Matrix<double>.Build.DenseOfRowArrays(Sigma);
            var evd = mat.Evd(MathNet.Numerics.LinearAlgebra.Symmetricity.Symmetric);

            double[] rawEigenvalues = evd.EigenValues.Real().ToArray();
            var rawVectors = evd.EigenVectors.ToArray();

            double[][] U_raw = new double[n][];
            for (int i = 0; i < n; i++)
            {
                U_raw[i] = new double[n];
                for (int j = 0; j < n; j++)
                    U_raw[i][j] = rawVectors[i, j];
            }

            // Sort by eigenvalue descending
            var pairs = rawEigenvalues
                .Select((val, idx) => new { val, vec = U_raw.Select(row => row[idx]).ToArray() })
                .OrderByDescending(p => p.val).ToList();

            double[] eigenvalues = pairs.Select(p => p.val).ToArray();
            double[][] U = new double[n][];
            for (int i = 0; i < n; i++) U[i] = new double[n];
            for (int j = 0; j < n; j++)
            {
                var col = pairs[j].vec;
                for (int i = 0; i < n; i++)
                    U[i][j] = col[i];
            }

            // Select components
            int kComp;
            double varianceKept;
            double total = eigenvalues.Sum();

            if (mode == 0) // variance threshold
            {
                double cum = 0;
                kComp = n;
                varianceKept = 1.0;
                for (int i = 0; i < eigenvalues.Length; i++)
                {
                    cum += eigenvalues[i];
                    if (cum / total >= value)
                    {
                        kComp = i + 1;
                        varianceKept = cum / total;
                        break;
                    }
                }
            }
            else // fixed number
            {
                kComp = Math.Max(1, Math.Min((int)value, n));
                varianceKept = eigenvalues.Take(kComp).Sum() / total;
            }

            // Reduced eigenvectors
            double[][] Ured = new double[n][];
            for (int i = 0; i < n; i++)
                Ured[i] = U[i].Take(kComp).ToArray();

            // Project
            double[][] Xproj = new double[m][];
            for (int i = 0; i < m; i++)
            {
                Xproj[i] = new double[kComp];
                for (int j = 0; j < kComp; j++)
                {
                    double sum = 0;
                    for (int f = 0; f < n; f++)
                        sum += Xnorm[i][f] * Ured[f][j];
                    Xproj[i][j] = sum;
                }
            }

            // Reconstruct
            double[][] Xrec = new double[m][];
            for (int i = 0; i < m; i++)
            {
                Xrec[i] = new double[n];
                for (int f = 0; f < n; f++)
                {
                    double sum = 0;
                    for (int j = 0; j < kComp; j++)
                        sum += Xproj[i][j] * Ured[f][j];
                    Xrec[i][f] = sum * std[f] + mu[f];
                }
            }

            // Explained variance ratio
            var evr = eigenvalues.Select(e => e / total).ToArray();
            var cumVar = new double[eigenvalues.Length];
            double cs = 0;
            for (int i = 0; i < eigenvalues.Length; i++)
            {
                cs += evr[i];
                cumVar[i] = cs;
            }

            return new PcaResult
            {
                FeatureMeans = mu,
                FeatureScales = std,
                ProjectedData = Xproj,
                ReconstructedData = Xrec,
                Eigenvectors = U,
                Eigenvalues = eigenvalues,
                CovarianceMatrix = Sigma,
                ComponentsKept = kComp,
                VarianceRetained = varianceKept * 100,
                ExplainedVarianceRatio = evr,
                CumulativeVariance = cumVar,
                OriginalHeaders = dataset.Headers.Take(featureCols).ToArray()
            };
        }
    }
}
