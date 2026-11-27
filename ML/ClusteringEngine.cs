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

    // Full agglomerative merge tree (SciPy-style): leaves are 0..N-1, merge i creates node N+i.
    public class Dendrogram
    {
        public int N { get; set; }
        public string Linkage { get; set; }
        public int[] Left { get; set; }
        public int[] Right { get; set; }
        public double[] Height { get; set; }   // non-decreasing for single/average/complete linkage
        public int[] Size { get; set; }
        public int[] LeafOrder { get; set; }   // left-to-right leaf order for drawing

        // Number of clusters left when the tree is cut just above `height`.
        public int ClustersAtHeight(double height) => N - Height.Count(h => h <= height);

        // Cut height that leaves exactly k clusters (midway between the two relevant merges).
        public double CutHeight(int k)
        {
            k = Math.Max(1, Math.Min(k, N));
            int merges = N - k;                       // merges applied
            double below = merges > 0 ? Height[merges - 1] : 0;
            double above = merges < Height.Length ? Height[merges] : below + Math.Max(1e-9, below * 0.05);
            return (below + above) / 2;
        }

        // Cluster label per leaf after applying the first N-k merges.
        public int[] Cut(int k)
        {
            k = Math.Max(1, Math.Min(k, N));
            var parent = Enumerable.Range(0, 2 * N - 1).ToArray();
            int Find(int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; }
            for (int i = 0; i < N - k; i++) { parent[Find(Left[i])] = N + i; parent[Find(Right[i])] = N + i; }
            var ids = new Dictionary<int, int>();
            var labels = new int[N];
            foreach (int leaf in LeafOrder.Length == N ? LeafOrder : Enumerable.Range(0, N).ToArray())
            {
                int root = Find(leaf);
                if (!ids.TryGetValue(root, out int id)) ids[root] = id = ids.Count;
                labels[leaf] = id;
            }
            return labels;
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

        // Optional tree from BuildDendrogram on the same data/linkage, reused by Hierarchical runs.
        public Dendrogram Precomputed { get; set; }

        // Largest row count for the hierarchical tree (O(n^2) memory, O(n^3) time).
        public const int MaxHierarchicalRows = 2000;

        // Column-wise standardization shared by training, the elbow curve and the dendrogram.
        public static double[][] Standardize(MLDataset ds, out double[] mean, out double[] std)
        {
            int n = ds.RowCount, m = ds.ColCount;
            mean = new double[m]; std = new double[m];
            for (int j = 0; j < m; j++)
            {
                double mu = 0; for (int i = 0; i < n; i++) mu += ds.Rows[i][j]; mu /= n;
                double s = 0; for (int i = 0; i < n; i++) { double d = ds.Rows[i][j] - mu; s += d * d; }
                s = Math.Sqrt(s / n); if (s == 0) s = 1e-8;
                mean[j] = mu; std[j] = s;
            }
            var Z = new double[n][];
            for (int i = 0; i < n; i++) { Z[i] = new double[m]; for (int j = 0; j < m; j++) Z[i][j] = (ds.Rows[i][j] - mean[j]) / std[j]; }
            return Z;
        }

        public ClusterResult Run(MLDataset ds)
        {
            int n = ds.RowCount;
            var Z = Standardize(ds, out var mean, out var std);

            int[] labels; double[][] centroids = null;
            switch (Type)
            {
                case ClustererType.KMeans: labels = KMeansRun(Z, K, Seed, out centroids); break;
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
        private int[] KMeansRun(double[][] Z, int kWanted, int seed, out double[][] centroids)
        {
            int n = Z.Length, m = Z[0].Length, k = Math.Max(1, Math.Min(kWanted, n));
            var rng = new Random(seed);
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

        // ---- Elbow method: within-cluster sum of squares for k = 1..kMax ----
        // Each k keeps the best of several K-Means starts so the curve is smooth and repeatable.
        public (int k, double wcss)[] ElbowCurve(MLDataset ds, int kMax, int restarts = 5)
        {
            var Z = Standardize(ds, out _, out _);
            int top = Math.Max(1, Math.Min(kMax, Z.Length));
            var curve = new (int, double)[top];
            for (int k = 1; k <= top; k++)
            {
                double best = double.MaxValue;
                for (int r = 0; r < restarts; r++)
                {
                    var labels = KMeansRun(Z, k, Seed + r, out var centroids);
                    double w = 0; for (int i = 0; i < Z.Length; i++) w += Dist2(Z[i], centroids[labels[i]]);
                    best = Math.Min(best, w);
                }
                curve[k - 1] = (k, best);
            }
            return curve;
        }

        // Suggested elbow: the point farthest from the straight line joining the curve's ends.
        public static int SuggestElbow((int k, double wcss)[] curve)
        {
            if (curve.Length < 3) return curve.Length > 0 ? curve[^1].k : 1;
            double x1 = curve[0].k, y1 = curve[0].wcss, x2 = curve[^1].k, y2 = curve[^1].wcss;
            double xr = x2 - x1, yr = Math.Abs(y1 - y2) < 1e-12 ? 1 : y1 - y2;
            int best = curve[0].k; double bd = -1;
            foreach (var (k, w) in curve)
            {
                double x = (k - x1) / xr, y = (y1 - w) / yr;     // normalized to the unit square
                double d = Math.Abs(y - x) / Math.Sqrt(2);
                if (d > bd) { bd = d; best = k; }
            }
            return best;
        }

        // ---- Agglomerative hierarchical: full tree via Lance-Williams updates ----
        public Dendrogram BuildDendrogram(MLDataset ds) => BuildDendrogram(Standardize(ds, out _, out _));

        private Dendrogram BuildDendrogram(double[][] Z)
        {
            int n = Z.Length;
            if (n > MaxHierarchicalRows)
                throw new InvalidOperationException($"Hierarchical clustering supports up to {MaxHierarchicalRows} rows; this dataset has {n}.");
            string link = Linkage == "single" || Linkage == "complete" ? Linkage : "average";
            var D = new double[n][];
            for (int i = 0; i < n; i++) { D[i] = new double[n]; for (int j = 0; j < i; j++) D[i][j] = D[j][i] = Math.Sqrt(Dist2(Z[i], Z[j])); }
            var active = new List<int>(Enumerable.Range(0, n));
            var nodeOf = Enumerable.Range(0, n).ToArray();   // row/col slot -> tree node id
            var size = Enumerable.Repeat(1, n).ToArray();
            int merges = Math.Max(0, n - 1);
            var tree = new Dendrogram { N = n, Linkage = link, Left = new int[merges], Right = new int[merges], Height = new double[merges], Size = new int[merges] };
            for (int step = 0; step < merges; step++)
            {
                int a = -1, b = -1; double bd = double.MaxValue;
                for (int x = 0; x < active.Count; x++)
                    for (int y = x + 1; y < active.Count; y++)
                    { double d = D[active[x]][active[y]]; if (d < bd) { bd = d; a = active[x]; b = active[y]; } }
                foreach (int k in active)
                {
                    if (k == a || k == b) continue;
                    double da = D[k][a], db = D[k][b];
                    double nd = link == "single" ? Math.Min(da, db) : link == "complete" ? Math.Max(da, db)
                        : (size[a] * da + size[b] * db) / (size[a] + size[b]);
                    D[k][a] = D[a][k] = nd;
                }
                tree.Left[step] = nodeOf[a]; tree.Right[step] = nodeOf[b];
                tree.Height[step] = bd; size[a] += size[b]; tree.Size[step] = size[a];
                nodeOf[a] = n + step; active.Remove(b);
            }
            // Leaf order: iterative depth-first walk from the root.
            var order = new List<int>(n);
            var stack = new Stack<int>(); if (n > 0) stack.Push(n > 1 ? 2 * n - 2 : 0);
            while (stack.Count > 0)
            {
                int node = stack.Pop();
                if (node < n) { order.Add(node); continue; }
                stack.Push(tree.Right[node - n]); stack.Push(tree.Left[node - n]);
            }
            tree.LeafOrder = order.ToArray();
            return tree;
        }

        private int[] HierarchicalRun(double[][] Z)
        {
            var tree = Precomputed != null && Precomputed.N == Z.Length ? Precomputed : BuildDendrogram(Z);
            return tree.Cut(K);
        }

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
