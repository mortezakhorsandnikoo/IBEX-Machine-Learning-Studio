using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace ArchiAI.ML
{
    public enum ClustererType { KMeans = 0, DBSCAN = 1, Hierarchical = 2, Fuzzy = 3 }

    public class ClusterResult
    {
        public ClustererType Type { get; set; }
        public int[] Labels { get; set; }          // cluster id per row (-1 = noise, DBSCAN)
        public int NumClusters { get; set; }
        public double Silhouette { get; set; }
        public int[] ClusterSizes { get; set; }
        public double[][] Proj2D { get; set; }      // n x 2 (PC1, PC2) for the scatter
        public string[] FeatureNames { get; set; }
        public string Info { get; set; }

        public double[] Means { get; set; }
        public double[] Stds { get; set; }
        public double[][] Centroids { get; set; }   // standardized (KMeans/Fuzzy); null otherwise
        public double[][] StdX { get; set; }        // standardized training rows (for 1-NN assign)

        public int Assign(double[] rawX)
        {
            var x = new double[rawX.Length];
            for (int j = 0; j < rawX.Length; j++)
            {
                double s = (Stds == null || Stds[j] == 0) ? 1e-10 : Stds[j];
                x[j] = (rawX[j] - (Means == null ? 0 : Means[j])) / s;
            }
            if (Centroids != null && Centroids.Length > 0)
            {
                int best = 0; double bd = double.MaxValue;
                for (int c = 0; c < Centroids.Length; c++)
                {
                    double d = 0; for (int j = 0; j < x.Length; j++) { double e = x[j] - Centroids[c][j]; d += e * e; }
                    if (d < bd) { bd = d; best = c; }
                }
                return best;
            }
            // 1-NN to a labeled training point (DBSCAN/Hierarchical)
            int bi = 0; double bb = double.MaxValue;
            for (int i = 0; i < StdX.Length; i++)
            {
                if (Labels[i] < 0) continue;
                double d = 0; for (int j = 0; j < x.Length; j++) { double e = x[j] - StdX[i][j]; d += e * e; }
                if (d < bb) { bb = d; bi = i; }
            }
            return Labels[bi];
        }
    }

    public class ClusteringEngine
    {
        public ClustererType Type { get; set; } = ClustererType.KMeans;
        public int K { get; set; } = 3;
        public int Iterations { get; set; } = 100;
        public double Eps { get; set; } = 0.5;
        public int MinPts { get; set; } = 5;
        public double Fuzziness { get; set; } = 2.0;
        public string Linkage { get; set; } = "average"; // "single" | "average" | "complete"
        public int Seed { get; set; } = 42;

        public ClusterResult Run(MLDataset ds)
        {
            int n = ds.RowCount, m = ds.ColCount;
            var X = new double[n][];
            for (int i = 0; i < n; i++) { X[i] = new double[m]; for (int j = 0; j < m; j++) X[i][j] = ds.Rows[i][j]; }
            // standardize
            var mean = new double[m]; var std = new double[m];
            for (int j = 0; j < m; j++)
            {
                double mu = 0; for (int i = 0; i < n; i++) mu += X[i][j]; mu /= n;
                double s = 0; for (int i = 0; i < n; i++) { double d = X[i][j] - mu; s += d * d; }
                s = Math.Sqrt(s / n); if (s == 0) s = 1e-8;
                mean[j] = mu; std[j] = s;
            }
            var Z = new double[n][];
            for (int i = 0; i < n; i++) { Z[i] = new double[m]; for (int j = 0; j < m; j++) Z[i][j] = (X[i][j] - mean[j]) / std[j]; }

            int[] labels; double[][] centroids = null;
            switch (Type)
            {
                case ClustererType.KMeans: labels = KMeansRun(Z, out centroids); break;
                case ClustererType.Fuzzy: labels = FuzzyRun(Z, out centroids); break;
                case ClustererType.DBSCAN: labels = DbscanRun(Z); break;
                default: labels = HierarchicalRun(Z); break;
            }

            var clusters = labels.Where(l => l >= 0).Distinct().OrderBy(v => v).ToArray();
            int nc = clusters.Length;
            var sizes = new int[nc];
            for (int i = 0; i < n; i++) if (labels[i] >= 0) sizes[Array.IndexOf(clusters, labels[i])]++;

            double sil = (nc >= 2) ? Silhouette(Z, labels) : 0.0;
            var proj = Pca2D(Z);

            return new ClusterResult
            {
                Type = Type, Labels = labels, NumClusters = nc, Silhouette = sil,
                ClusterSizes = sizes, Proj2D = proj, FeatureNames = ds.Headers.ToArray(),
                Means = mean, Stds = std, Centroids = centroids, StdX = Z,
                Info = $"{Type}: {nc} clusters" + (labels.Any(l => l < 0) ? $", {labels.Count(l => l < 0)} noise pts" : "")
            };
        }

        private static double Dist2(double[] a, double[] b) { double s = 0; for (int j = 0; j < a.Length; j++) { double d = a[j] - b[j]; s += d * d; } return s; }

        // ---- K-Means (Lloyd) ----
        private int[] KMeansRun(double[][] Z, out double[][] centroids)
        {
            int n = Z.Length, m = Z[0].Length, k = Math.Max(1, Math.Min(K, n));
            var rng = new Random(Seed);
            centroids = new double[k][];
            var chosen = Enumerable.Range(0, n).OrderBy(_ => rng.Next()).Take(k).ToArray();
            for (int c = 0; c < k; c++) centroids[c] = (double[])Z[chosen[c]].Clone();
            var labels = new int[n];
            for (int it = 0; it < Iterations; it++)
            {
                bool changed = false;
                for (int i = 0; i < n; i++)
                {
                    int best = 0; double bd = double.MaxValue;
                    for (int c = 0; c < k; c++) { double d = Dist2(Z[i], centroids[c]); if (d < bd) { bd = d; best = c; } }
                    if (labels[i] != best) { labels[i] = best; changed = true; }
                }
                var sum = new double[k][]; var cnt = new int[k];
                for (int c = 0; c < k; c++) sum[c] = new double[m];
                for (int i = 0; i < n; i++) { cnt[labels[i]]++; for (int j = 0; j < m; j++) sum[labels[i]][j] += Z[i][j]; }
                for (int c = 0; c < k; c++) if (cnt[c] > 0) for (int j = 0; j < m; j++) centroids[c][j] = sum[c][j] / cnt[c];
                if (!changed) break;
            }
            return labels;
        }

        // ---- Fuzzy C-Means (hard labels = argmax membership) ----
        private int[] FuzzyRun(double[][] Z, out double[][] centroids)
        {
            int n = Z.Length, m = Z[0].Length, k = Math.Max(1, Math.Min(K, n));
            double mm = Math.Max(1.01, Fuzziness);
            var rng = new Random(Seed);
            var U = new double[n][];
            for (int i = 0; i < n; i++) { U[i] = new double[k]; double s = 0; for (int c = 0; c < k; c++) { U[i][c] = rng.NextDouble(); s += U[i][c]; } for (int c = 0; c < k; c++) U[i][c] /= s; }
            centroids = new double[k][]; for (int c = 0; c < k; c++) centroids[c] = new double[m];
            for (int it = 0; it < Iterations; it++)
            {
                for (int c = 0; c < k; c++)
                {
                    var num = new double[m]; double den = 0;
                    for (int i = 0; i < n; i++) { double w = Math.Pow(U[i][c], mm); den += w; for (int j = 0; j < m; j++) num[j] += w * Z[i][j]; }
                    if (den < 1e-12) den = 1e-12;
                    for (int j = 0; j < m; j++) centroids[c][j] = num[j] / den;
                }
                for (int i = 0; i < n; i++)
                {
                    for (int c = 0; c < k; c++)
                    {
                        double dic = Math.Sqrt(Dist2(Z[i], centroids[c])) + 1e-12, s = 0;
                        for (int c2 = 0; c2 < k; c2++) { double d2 = Math.Sqrt(Dist2(Z[i], centroids[c2])) + 1e-12; s += Math.Pow(dic / d2, 2.0 / (mm - 1)); }
                        U[i][c] = 1.0 / s;
                    }
                }
            }
            var labels = new int[n];
            for (int i = 0; i < n; i++) { int bi = 0; for (int c = 1; c < k; c++) if (U[i][c] > U[i][bi]) bi = c; labels[i] = bi; }
            return labels;
        }

        // ---- DBSCAN ----
        private int[] DbscanRun(double[][] Z)
        {
            int n = Z.Length; var labels = new int[n]; for (int i = 0; i < n; i++) labels[i] = -2; // unvisited
            double eps2 = Eps * Eps; int cid = -1;
            List<int> Region(int p) { var r = new List<int>(); for (int q = 0; q < n; q++) if (Dist2(Z[p], Z[q]) <= eps2) r.Add(q); return r; }
            for (int p = 0; p < n; p++)
            {
                if (labels[p] != -2) continue;
                var neigh = Region(p);
                if (neigh.Count < MinPts) { labels[p] = -1; continue; }
                cid++; labels[p] = cid;
                var seeds = new Queue<int>(neigh.Where(q => q != p));
                while (seeds.Count > 0)
                {
                    int q = seeds.Dequeue();
                    if (labels[q] == -1) labels[q] = cid;
                    if (labels[q] != -2) continue;
                    labels[q] = cid;
                    var qn = Region(q);
                    if (qn.Count >= MinPts) foreach (var x in qn) if (labels[x] == -2 || labels[x] == -1) seeds.Enqueue(x);
                }
            }
            for (int i = 0; i < n; i++) if (labels[i] == -2) labels[i] = -1;
            return labels;
        }

        // ---- Agglomerative hierarchical (until K clusters) ----
        private int[] HierarchicalRun(double[][] Z)
        {
            int n = Z.Length, k = Math.Max(1, Math.Min(K, n));
            var members = new List<List<int>>();
            for (int i = 0; i < n; i++) members.Add(new List<int> { i });
            double ClusterDist(List<int> a, List<int> b)
            {
                double best = Linkage == "complete" ? double.MinValue : double.MaxValue, sum = 0; int cc = 0;
                foreach (var i in a) foreach (var j in b)
                {
                    double d = Math.Sqrt(Dist2(Z[i], Z[j]));
                    if (Linkage == "single") best = Math.Min(best, d);
                    else if (Linkage == "complete") best = Math.Max(best, d);
                    else { sum += d; cc++; }
                }
                return Linkage == "average" ? (cc > 0 ? sum / cc : 0) : best;
            }
            while (members.Count > k)
            {
                int ai = 0, bi = 1; double bd = double.MaxValue;
                for (int i = 0; i < members.Count; i++)
                    for (int j = i + 1; j < members.Count; j++)
                    { double d = ClusterDist(members[i], members[j]); if (d < bd) { bd = d; ai = i; bj_set(ref bi, j); } }
                members[ai].AddRange(members[bi]); members.RemoveAt(bi);
            }
            var labels = new int[n];
            for (int c = 0; c < members.Count; c++) foreach (var i in members[c]) labels[i] = c;
            return labels;
        }
        private static void bj_set(ref int bi, int j) { bi = j; }

        // ---- silhouette (euclidean, standardized) ----
        private static double Silhouette(double[][] Z, int[] labels)
        {
            int n = Z.Length;
            var clusters = labels.Where(l => l >= 0).Distinct().ToArray();
            var byCluster = clusters.ToDictionary(c => c, c => Enumerable.Range(0, n).Where(i => labels[i] == c).ToArray());
            double total = 0; int cnt = 0;
            for (int i = 0; i < n; i++)
            {
                if (labels[i] < 0) continue;
                var own = byCluster[labels[i]];
                if (own.Length <= 1) { continue; }
                double a = own.Where(j => j != i).Select(j => Math.Sqrt(Dist2(Z[i], Z[j]))).Average();
                double b = double.MaxValue;
                foreach (var c in clusters)
                {
                    if (c == labels[i]) continue;
                    double mean = byCluster[c].Select(j => Math.Sqrt(Dist2(Z[i], Z[j]))).Average();
                    if (mean < b) b = mean;
                }
                total += (b - a) / Math.Max(a, b); cnt++;
            }
            return cnt > 0 ? total / cnt : 0;
        }

        // ---- PCA to 2D for the scatter ----
        private static double[][] Pca2D(double[][] Z)
        {
            int n = Z.Length, m = Z[0].Length;
            var cov = Matrix<double>.Build.Dense(m, m);
            for (int a = 0; a < m; a++)
                for (int b = 0; b < m; b++)
                {
                    double s = 0; for (int i = 0; i < n; i++) s += Z[i][a] * Z[i][b];
                    cov[a, b] = s / n;
                }
            var evd = cov.Evd(MathNet.Numerics.LinearAlgebra.Symmetricity.Symmetric);
            var vals = evd.EigenValues.Real().ToArray();
            var vecs = evd.EigenVectors;
            var order = Enumerable.Range(0, m).OrderByDescending(i => vals[i]).ToArray();
            int p1 = order[0], p2 = m > 1 ? order[1] : order[0];
            var proj = new double[n][];
            for (int i = 0; i < n; i++)
            {
                double a = 0, b = 0;
                for (int j = 0; j < m; j++) { a += Z[i][j] * vecs[j, p1]; b += Z[i][j] * vecs[j, p2]; }
                proj[i] = new[] { a, b };
            }
            return proj;
        }
    }
}
