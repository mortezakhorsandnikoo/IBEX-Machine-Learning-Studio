// Requires NuGet: XGBoostSharp  (high-level XGBRegressor)
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using XGBoostSharp;

namespace ArchiAI.ML
{
    public class XGBoostModel : IRegressionModel
    {
        private XGBRegressor _model;
        private int _numRounds = 200, _maxDepth = 6;
        private float _eta = 0.1f;

        public string Name => "XGBoost";
        public Dictionary<string, double> GetDefaultHyperparameters() => new Dictionary<string, double>
        { { "numRounds", 200 }, { "maxDepth", 6 }, { "eta", 0.1 }, { "subsample", 0.8 } };
        public void SetHyperparameters(Dictionary<string, double> p)
        {
            if (p.ContainsKey("numRounds")) _numRounds = Math.Max(10, (int)Math.Round(p["numRounds"]));
            if (p.ContainsKey("maxDepth")) _maxDepth = Math.Max(1, (int)Math.Round(p["maxDepth"]));
            if (p.ContainsKey("eta")) _eta = (float)Math.Max(0.001, Math.Min(1.0, p["eta"]));
        }
        public void Train(Dataset data)
        {
            _model?.Dispose();
            _model = new XGBRegressor(maxDepth: _maxDepth, learningRate: _eta, nEstimators: _numRounds);
            _model.Fit(data.X, data.y);
        }
        public float[] Predict(float[][] X)
        {
            if (_model == null) throw new InvalidOperationException("XGBoost not trained.");
            return _model.Predict(X);
        }
        public string SaveState()
        {
            var tmp = Path.GetTempFileName();
            try { _model.SaveModelToFile(tmp); return Convert.ToBase64String(File.ReadAllBytes(tmp)); }
            finally { try { File.Delete(tmp); } catch { } }
        }
        public void LoadState(string s)
        {
            var tmp = Path.GetTempFileName();
            try { File.WriteAllBytes(tmp, Convert.FromBase64String(s)); _model = XGBRegressor.LoadFromFile(tmp); }
            finally { try { File.Delete(tmp); } catch { } }
        }
        public void Dispose() { _model?.Dispose(); _model = null; }
    }
}
