using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using MathNet.Numerics.LinearAlgebra;

namespace ArchiAI.ML
{
    // Radial Basis Function surrogate (Gaussian basis at each training point).
    public class RbfModel : IRegressionModel
    {
        private float[][] _centers;
        private double[] _weights;
        private double _epsilon = 1.0, _smoothing = 0.0;
        private bool _autoEpsilon = true;

        public string Name => "RBF";
        public Dictionary<string, double> GetDefaultHyperparameters() => new Dictionary<string, double>
        { { "epsilon", 1.0 }, { "smoothing", 0.0 } };
        public void SetHyperparameters(Dictionary<string, double> p)
        {
            if (p.ContainsKey("epsilon")) { double e = p["epsilon"]; if (e <= 0) _autoEpsilon = true; else { _autoEpsilon = false; _epsilon = e; } }
            if (p.ContainsKey("smoothing")) _smoothing = Math.Max(0, p["smoothing"]);
        }
        private double Phi(double r) { double er = _epsilon * r; return Math.Exp(-(er * er)); }
        private static double Dist(float[] a, float[] b) { double s = 0; for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; s += d * d; } return Math.Sqrt(s); }

        public void Train(Dataset data)
        {
            int n = data.X.Length;
            if (n == 0) throw new Exception("RBF: no training data.");
            _centers = new float[n][];
            for (int i = 0; i < n; i++) _centers[i] = (float[])data.X[i].Clone();
            if (_autoEpsilon)
            {
                double mn = MeanNn(_centers); if (mn < 1e-9) mn = 1.0; _epsilon = 1.0 / mn;
            }
            var phi = Matrix<double>.Build.Dense(n, n);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++) phi[i, j] = Phi(Dist(_centers[i], _centers[j]));
                phi[i, i] += _smoothing;
            }
            var y = Vector<double>.Build.Dense(n, i => data.y[i]);
            Vector<double> w;
            try { w = phi.Solve(y); if (w.Any(v => double.IsNaN(v) || double.IsInfinity(v))) w = LstSq(phi, y); }
            catch { w = LstSq(phi, y); }
            _weights = w.ToArray();
        }
        private Vector<double> LstSq(Matrix<double> a, Vector<double> b)
        {
            var at = a.Transpose(); var ata = at * a;
            for (int i = 0; i < ata.RowCount; i++) ata[i, i] += 1e-8;
            return ata.Solve(at * b);
        }
        public float[] Predict(float[][] X)
        {
            if (_centers == null) throw new InvalidOperationException("RBF not trained.");
            var o = new float[X.Length];
            int n = _centers.Length;
            for (int k = 0; k < X.Length; k++)
            {
                double s = 0; for (int i = 0; i < n; i++) s += _weights[i] * Phi(Dist(X[k], _centers[i]));
                o[k] = (float)s;
            }
            return o;
        }
        private static double MeanNn(float[][] pts)
        {
            int n = pts.Length; if (n < 2) return 1.0;
            double total = 0; int c = 0;
            for (int i = 0; i < n; i++)
            {
                double best = double.MaxValue;
                for (int j = 0; j < n; j++) { if (i == j) continue; double d = Dist(pts[i], pts[j]); if (d < best) best = d; }
                if (best < double.MaxValue) { total += best; c++; }
            }
            return c > 0 ? total / c : 1.0;
        }
        private class St { public float[][] centers; public double[] weights; public double epsilon; public double smoothing; }
        public string SaveState() => JsonSerializer.Serialize(new St { centers = _centers, weights = _weights, epsilon = _epsilon, smoothing = _smoothing }, RegModelIO.J);
        public void LoadState(string s) { var st = JsonSerializer.Deserialize<St>(s, RegModelIO.J); _centers = st.centers; _weights = st.weights; _epsilon = st.epsilon; _smoothing = st.smoothing; _autoEpsilon = false; }
        public void Dispose() { }
    }
}