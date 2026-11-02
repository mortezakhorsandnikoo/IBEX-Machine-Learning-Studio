using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace ArchiAI.ML
{
    public enum ActivationType { ReLU = 0, Tanh = 1, Sigmoid = 2, Linear = 3 }
    public enum OptimizerType { SGD = 0, Adam = 1 }

    // ---- shared standardizer ----
    internal static class Std
    {
        public static void Fit(float[][] X, out double[] mean, out double[] std)
        {
            int n = X.Length, m = X[0].Length;
            mean = new double[m]; std = new double[m];
            for (int j = 0; j < m; j++)
            {
                double mu = 0; for (int i = 0; i < n; i++) mu += X[i][j]; mu /= n;
                double s = 0; for (int i = 0; i < n; i++) { double d = X[i][j] - mu; s += d * d; }
                s = Math.Sqrt(s / n); if (s == 0) s = 1e-8;
                mean[j] = mu; std[j] = s;
            }
        }
        public static double[] Apply(float[] x, double[] mean, double[] std)
        {
            var o = new double[x.Length];
            for (int j = 0; j < x.Length; j++) o[j] = (x[j] - mean[j]) / std[j];
            return o;
        }
    }

    // ═══════════════ NORMAL EQUATION (closed-form, ridge, standardized) ═══════════════
    public class NormalEquationModel : IRegressionModel
    {
        private double[] _mean, _std;
        private Vector<double> _theta; // [bias, w1..wn]
        private double _ridge = 0.0;
        public string Name => "Normal Equation";
        public Dictionary<string, double> GetDefaultHyperparameters() => new Dictionary<string, double> { { "ridge", 0.0 } };
        public void SetHyperparameters(Dictionary<string, double> p) { if (p.ContainsKey("ridge")) _ridge = Math.Max(0, p["ridge"]); }

        public void Train(Dataset data)
        {
            int n = data.X.Length, m = data.FeatureCount;
            Std.Fit(data.X, out _mean, out _std);
            var Xa = Matrix<double>.Build.Dense(n, m + 1);
            var y = Vector<double>.Build.Dense(n);
            for (int i = 0; i < n; i++)
            {
                Xa[i, 0] = 1.0;
                var xs = Std.Apply(data.X[i], _mean, _std);
                for (int j = 0; j < m; j++) Xa[i, j + 1] = xs[j];
                y[i] = data.y[i];
            }
            var Xt = Xa.Transpose();
            var A = Xt * Xa;
            for (int i = 0; i < A.RowCount; i++) A[i, i] += _ridge + 1e-8; // ridge + stability
            _theta = A.Solve(Xt * y);
        }
        public float[] Predict(float[][] X)
        {
            var o = new float[X.Length];
            for (int i = 0; i < X.Length; i++)
            {
                var xs = Std.Apply(X[i], _mean, _std);
                double v = _theta[0];
                for (int j = 0; j < xs.Length; j++) v += _theta[j + 1] * xs[j];
                o[i] = (float)v;
            }
            return o;
        }
        private class St { public double[] mean; public double[] std; public double[] theta; public double ridge; }
        public string SaveState() => JsonSerializer.Serialize(new St { mean = _mean, std = _std, theta = _theta?.ToArray(), ridge = _ridge }, RegModelIO.J);
        public void LoadState(string s) { var st = JsonSerializer.Deserialize<St>(s, RegModelIO.J); _mean = st.mean; _std = st.std; _theta = Vector<double>.Build.DenseOfArray(st.theta); _ridge = st.ridge; }
        public void Dispose() { }
    }

    // ═══════════════ LOCALLY WEIGHTED REGRESSION ═══════════════
    public class LwrModel : IRegressionModel
    {
        private double[] _mean, _std;
        private double[][] _Xn; private double[] _y;
        private double _tau = 1.0, _lambda = 1e-3;
        public string Name => "LWR";
        public Dictionary<string, double> GetDefaultHyperparameters() => new Dictionary<string, double> { { "tau", 1.0 }, { "lambda", 1e-3 } };
        public void SetHyperparameters(Dictionary<string, double> p)
        {
            if (p.ContainsKey("tau")) _tau = Math.Max(1e-3, p["tau"]);
            if (p.ContainsKey("lambda")) _lambda = Math.Max(0, p["lambda"]);
        }
        public void Train(Dataset data)
        {
            Std.Fit(data.X, out _mean, out _std);
            _Xn = new double[data.X.Length][];
            for (int i = 0; i < data.X.Length; i++) _Xn[i] = Std.Apply(data.X[i], _mean, _std);
            _y = data.y.Select(v => (double)v).ToArray();
        }
        public float[] Predict(float[][] X)
        {
            int m = _Xn.Length, d = _Xn.Length > 0 ? _Xn[0].Length : 0;
            var o = new float[X.Length];
            for (int q = 0; q < X.Length; q++)
            {
                var xq = Std.Apply(X[q], _mean, _std);
                // weighted ridge least squares with intercept
                var A = Matrix<double>.Build.Dense(d + 1, d + 1);
                var bvec = Vector<double>.Build.Dense(d + 1);
                for (int i = 0; i < m; i++)
                {
                    double dist2 = 0; for (int k = 0; k < d; k++) { double df = xq[k] - _Xn[i][k]; dist2 += df * df; }
                    double w = Math.Exp(-dist2 / (2 * _tau * _tau));
                    var xi = new double[d + 1]; xi[0] = 1.0;
                    for (int k = 0; k < d; k++) xi[k + 1] = _Xn[i][k];
                    for (int a = 0; a < d + 1; a++)
                    {
                        for (int b = 0; b < d + 1; b++) A[a, b] += w * xi[a] * xi[b];
                        bvec[a] += w * xi[a] * _y[i];
                    }
                }
                for (int a = 0; a < d + 1; a++) A[a, a] += _lambda + 1e-8;
                var theta = A.Solve(bvec);
                double pred = theta[0];
                for (int k = 0; k < d; k++) pred += theta[k + 1] * xq[k];
                o[q] = (float)pred;
            }
            return o;
        }
        private class St { public double[] mean; public double[] std; public double[][] Xn; public double[] y; public double tau; public double lambda; }
        public string SaveState() => JsonSerializer.Serialize(new St { mean = _mean, std = _std, Xn = _Xn, y = _y, tau = _tau, lambda = _lambda }, RegModelIO.J);
        public void LoadState(string s) { var st = JsonSerializer.Deserialize<St>(s, RegModelIO.J); _mean = st.mean; _std = st.std; _Xn = st.Xn; _y = st.y; _tau = st.tau; _lambda = st.lambda; }
        public void Dispose() { }
    }

    // ═══════════════ RANDOM FOREST (keeps tree structure for the diagram) ═══════════════
    public class RegTreeNode
    {
        public bool IsLeaf; public double Value; public int Feature; public double Threshold;
        public RegTreeNode Left, Right;
    }
    public class RandomForestModel : IRegressionModel
    {
        private int _trees = 50, _maxDepth = 12, _minLeaf = 10, _seed = 42;
        public RegTreeNode[] Forest { get; private set; }
        public RegTreeNode FirstTree => (Forest != null && Forest.Length > 0) ? Forest[0] : null;
        public string Name => "Random Forest";
        public Dictionary<string, double> GetDefaultHyperparameters() => new Dictionary<string, double>
        { { "trees", 50 }, { "maxDepth", 12 }, { "minLeaf", 10 } };
        public void SetHyperparameters(Dictionary<string, double> p)
        {
            if (p.ContainsKey("trees")) _trees = Math.Max(1, (int)Math.Round(p["trees"]));
            if (p.ContainsKey("maxDepth")) _maxDepth = Math.Max(1, (int)Math.Round(p["maxDepth"]));
            if (p.ContainsKey("minLeaf")) _minLeaf = Math.Max(1, (int)Math.Round(p["minLeaf"]));
        }
        public void Train(Dataset data)
        {
            var X = data.X; var y = data.y;
            Forest = new RegTreeNode[_trees];
            int md = _maxDepth <= 0 ? int.MaxValue : _maxDepth;
            Parallel.For(0, _trees, t =>
            {
                var rng = new Random(_seed + t * 9973);
                int n = X.Length;
                var bx = new float[n][]; var by = new float[n];
                for (int i = 0; i < n; i++) { int id = rng.Next(n); bx[i] = X[id]; by[i] = y[id]; }
                Forest[t] = Build(bx, by, 0, md, rng);
            });
        }
        private RegTreeNode Build(float[][] X, float[] y, int depth, int maxDepth, Random rng)
        {
            if (depth >= maxDepth || y.Length <= _minLeaf)
                return new RegTreeNode { IsLeaf = true, Value = y.Length > 0 ? y.Average() : 0 };
            int nf = X[0].Length;
            int k = Math.Max(1, (int)Math.Sqrt(nf));
            var feats = Enumerable.Range(0, nf).OrderBy(_ => rng.Next()).Take(k);
            int bestF = -1; double bestT = 0, bestLoss = double.MaxValue; bool found = false;
            foreach (int f in feats)
            {
                var vals = X.Select(r => (double)r[f]).OrderBy(v => v).ToArray();
                int step = Math.Max(1, vals.Length / 20);
                for (int i = 0; i < vals.Length - 1; i += step)
                {
                    double thr = 0.5 * (vals[i] + vals[i + 1]);
                    var lY = new List<float>(); var rY = new List<float>();
                    for (int r = 0; r < X.Length; r++) { if (X[r][f] < thr) lY.Add(y[r]); else rY.Add(y[r]); }
                    if (lY.Count < _minLeaf || rY.Count < _minLeaf) continue;
                    double loss = Var(lY) + Var(rY);
                    if (loss < bestLoss) { bestLoss = loss; bestF = f; bestT = thr; found = true; }
                }
            }
            if (!found) return new RegTreeNode { IsLeaf = true, Value = y.Average() };
            var lx = new List<float[]>(); var lyy = new List<float>();
            var rx = new List<float[]>(); var ryy = new List<float>();
            for (int r = 0; r < X.Length; r++)
            {
                if (X[r][bestF] < bestT) { lx.Add(X[r]); lyy.Add(y[r]); }
                else { rx.Add(X[r]); ryy.Add(y[r]); }
            }
            return new RegTreeNode
            {
                IsLeaf = false,
                Feature = bestF,
                Threshold = bestT,
                Left = Build(lx.ToArray(), lyy.ToArray(), depth + 1, maxDepth, rng),
                Right = Build(rx.ToArray(), ryy.ToArray(), depth + 1, maxDepth, rng)
            };
        }
        private static double Var(List<float> y)
        {
            if (y.Count == 0) return 0; double m = y.Average(), s = 0;
            foreach (var v in y) { double d = v - m; s += d * d; }
            return s / y.Count;
        }
        private static double PredOne(RegTreeNode n, float[] x)
        {
            while (!n.IsLeaf) n = x[n.Feature] < n.Threshold ? n.Left : n.Right;
            return n.Value;
        }
        public float[] Predict(float[][] X)
        {
            var o = new float[X.Length];
            for (int i = 0; i < X.Length; i++)
            {
                double s = 0; foreach (var t in Forest) s += PredOne(t, X[i]);
                o[i] = (float)(s / Forest.Length);
            }
            return o;
        }
        private class St { public int trees; public int maxDepth; public int minLeaf; public int seed; public RegTreeNode[] forest; }
        public string SaveState() => JsonSerializer.Serialize(new St { trees = _trees, maxDepth = _maxDepth, minLeaf = _minLeaf, seed = _seed, forest = Forest }, RegModelIO.J);
        public void LoadState(string s) { var st = JsonSerializer.Deserialize<St>(s, RegModelIO.J); _trees = st.trees; _maxDepth = st.maxDepth; _minLeaf = st.minLeaf; _seed = st.seed; Forest = st.forest; }
        public void Dispose() { }
    }

    // ═══════════════ MLP (feed-forward neural net) ═══════════════
    public class MlpModel : IRegressionModel
    {
        private double[] _mean, _std;
        private int _hiddenSize = 32, _hiddenLayers = 1, _epochs = 200;
        private double _lr = 1e-3, _l2 = 0.0;
        private ActivationType _act = ActivationType.ReLU;
        private double[][][] W; private double[][] b; private int _in, _total;
        private readonly Random _rnd = new Random(0);

        public string Name => "MLP";
        public Dictionary<string, double> GetDefaultHyperparameters() => new Dictionary<string, double>
        { { "hiddenSize", 32 }, { "hiddenLayers", 1 }, { "epochs", 200 }, { "learningRate", 1e-3 } };
        public void SetHyperparameters(Dictionary<string, double> p)
        {
            if (p.ContainsKey("hiddenSize")) _hiddenSize = Math.Max(1, (int)Math.Round(p["hiddenSize"]));
            if (p.ContainsKey("hiddenLayers")) _hiddenLayers = Math.Max(0, (int)Math.Round(p["hiddenLayers"]));
            if (p.ContainsKey("epochs")) _epochs = Math.Max(1, (int)Math.Round(p["epochs"]));
            if (p.ContainsKey("learningRate")) _lr = Math.Max(1e-6, p["learningRate"]);
        }
        private double Act(double z) { switch (_act) { case ActivationType.ReLU: return z > 0 ? z : 0; case ActivationType.Tanh: return Math.Tanh(z); case ActivationType.Sigmoid: return 1 / (1 + Math.Exp(-z)); default: return z; } }
        private double ActD(double z) { switch (_act) { case ActivationType.ReLU: return z > 0 ? 1 : 0; case ActivationType.Tanh: double t = Math.Tanh(z); return 1 - t * t; case ActivationType.Sigmoid: double s = 1 / (1 + Math.Exp(-z)); return s * (1 - s); default: return 1; } }
        private double Nrm() { double u1 = 1 - _rnd.NextDouble(), u2 = 1 - _rnd.NextDouble(); return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2); }

        public void Train(Dataset data)
        {
            Std.Fit(data.X, out _mean, out _std);
            int n = data.X.Length; _in = data.FeatureCount;
            var X = new double[n][];
            for (int i = 0; i < n; i++) X[i] = Std.Apply(data.X[i], _mean, _std);
            var y = data.y;

            _total = _hiddenLayers + 1;
            W = new double[_total][][]; b = new double[_total][];
            for (int l = 0; l < _total; l++)
            {
                int inS = l == 0 ? _in : _hiddenSize;
                int outS = l == _total - 1 ? 1 : _hiddenSize;
                W[l] = new double[inS][]; for (int i = 0; i < inS; i++) W[l][i] = new double[outS];
                b[l] = new double[outS];
                double sd = Math.Sqrt(2.0 / (inS + outS));
                for (int i = 0; i < inS; i++) for (int j = 0; j < outS; j++) W[l][i][j] = sd * Nrm();
            }
            // simple full-batch SGD
            for (int ep = 0; ep < _epochs; ep++)
            {
                // forward store
                var A = new double[_total + 1][][]; var Z = new double[_total][][];
                A[0] = X;
                for (int l = 0; l < _total; l++)
                {
                    int inS = W[l].Length, outS = W[l][0].Length;
                    var z = new double[n][]; var a = new double[n][];
                    for (int i = 0; i < n; i++)
                    {
                        z[i] = new double[outS]; a[i] = new double[outS];
                        for (int j = 0; j < outS; j++)
                        {
                            double s = b[l][j];
                            for (int k = 0; k < inS; k++) s += A[l][i][k] * W[l][k][j];
                            z[i][j] = s;
                            a[i][j] = (l == _total - 1) ? s : Act(s);
                        }
                    }
                    Z[l] = z; A[l + 1] = a;
                }
                // backward
                var dZ = new double[n][];
                for (int i = 0; i < n; i++) { dZ[i] = new double[1]; dZ[i][0] = (A[_total][i][0] - y[i]) / n; }
                for (int l = _total - 1; l >= 0; l--)
                {
                    int inS = W[l].Length, outS = W[l][0].Length;
                    var Aprev = A[l];
                    var dW = new double[inS][]; for (int i = 0; i < inS; i++) dW[i] = new double[outS];
                    var db = new double[outS];
                    for (int i = 0; i < inS; i++)
                        for (int j = 0; j < outS; j++)
                        {
                            double s = 0; for (int r = 0; r < n; r++) s += Aprev[r][i] * dZ[r][j];
                            dW[i][j] = s + _l2 * W[l][i][j] / n;
                        }
                    for (int j = 0; j < outS; j++) { double s = 0; for (int r = 0; r < n; r++) s += dZ[r][j]; db[j] = s; }
                    if (l > 0)
                    {
                        int prev = inS;
                        var dZp = new double[n][];
                        for (int r = 0; r < n; r++)
                        {
                            dZp[r] = new double[prev];
                            for (int k = 0; k < prev; k++)
                            {
                                double s = 0; for (int j = 0; j < outS; j++) s += dZ[r][j] * W[l][k][j];
                                dZp[r][k] = s * ActD(Z[l - 1][r][k]);
                            }
                        }
                        dZ = dZp;
                    }
                    for (int i = 0; i < inS; i++) for (int j = 0; j < outS; j++) W[l][i][j] -= _lr * dW[i][j];
                    for (int j = 0; j < outS; j++) b[l][j] -= _lr * db[j];
                }
            }
        }
        public float[] Predict(float[][] X)
        {
            var o = new float[X.Length];
            for (int i = 0; i < X.Length; i++)
            {
                double[] a = Std.Apply(X[i], _mean, _std);
                for (int l = 0; l < _total; l++)
                {
                    int inS = W[l].Length, outS = W[l][0].Length;
                    var na = new double[outS];
                    for (int j = 0; j < outS; j++)
                    {
                        double s = b[l][j];
                        for (int k = 0; k < inS; k++) s += a[k] * W[l][k][j];
                        na[j] = (l == _total - 1) ? s : Act(s);
                    }
                    a = na;
                }
                o[i] = (float)a[0];
            }
            return o;
        }
        private class St { public double[] mean; public double[] std; public double[][][] W; public double[][] b; public int inp; public int total; public int hidden; public int layers; public int act; }
        public string SaveState() => JsonSerializer.Serialize(new St { mean = _mean, std = _std, W = W, b = b, inp = _in, total = _total, hidden = _hiddenSize, layers = _hiddenLayers, act = (int)_act }, RegModelIO.J);
        public void LoadState(string s)
        {
            var st = JsonSerializer.Deserialize<St>(s, RegModelIO.J);
            _mean = st.mean; _std = st.std; W = st.W; b = st.b;
            _in = st.inp; _total = st.total; _hiddenSize = st.hidden; _hiddenLayers = st.layers; _act = (ActivationType)st.act;
        }
        public void Dispose() { }
    }
}