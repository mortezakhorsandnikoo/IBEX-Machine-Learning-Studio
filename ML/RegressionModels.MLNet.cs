// Requires NuGet: Microsoft.ML  and  Microsoft.ML.LightGbm
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;

namespace ArchiAI.ML
{
    // Converts our float Dataset into an ML.NET IDataView.
    internal class MLNetAdapter
    {
        private readonly MLContext _ml;
        public MLNetAdapter(MLContext ml) { _ml = ml; }

        public IDataView ToDataView(Dataset data)
        {
            var list = new List<ModelInput>(data.X.Length);
            for (int i = 0; i < data.X.Length; i++)
                list.Add(new ModelInput { Features = data.X[i], Label = data.y[i] });
            var schema = SchemaDefinition.Create(typeof(ModelInput));
            schema["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, data.FeatureCount);
            schema["Label"].ColumnType = NumberDataViewType.Single;
            return _ml.Data.LoadFromEnumerable(list, schema);
        }
        public class ModelInput { public float[] Features { get; set; } public float Label { get; set; } }
        public class ModelOutput { public float Score { get; set; } }
    }

    // ═══════════════ FASTTREE ═══════════════
    public class FastTreeModel : IRegressionModel
    {
        private readonly MLContext _ml = new MLContext(seed: 42);
        private readonly MLNetAdapter _adapter;
        private ITransformer _model;
        private DataViewSchema _schema;
        private int _trees = 100, _leaves = 31, _minLeaf = 10;
        private double _lr = 0.2;
        public FastTreeModel() { _adapter = new MLNetAdapter(_ml); }
        public string Name => "FastTree";
        public Dictionary<string, double> GetDefaultHyperparameters() => new Dictionary<string, double>
        { { "numberOfTrees", 100 }, { "numberOfLeaves", 31 }, { "learningRate", 0.2 }, { "minExamplesPerLeaf", 10 } };
        public void SetHyperparameters(Dictionary<string, double> p)
        {
            if (p.ContainsKey("numberOfTrees")) _trees = Math.Max(10, (int)Math.Round(p["numberOfTrees"]));
            if (p.ContainsKey("numberOfLeaves")) _leaves = Math.Max(2, (int)Math.Round(p["numberOfLeaves"]));
            if (p.ContainsKey("learningRate")) _lr = Math.Max(0.001, Math.Min(1.0, p["learningRate"]));
            if (p.ContainsKey("minExamplesPerLeaf")) _minLeaf = Math.Max(1, (int)Math.Round(p["minExamplesPerLeaf"]));
        }
        public void Train(Dataset data)
        {
            var dv = _adapter.ToDataView(data);
            _schema = dv.Schema;
            var pipe = _ml.Regression.Trainers.FastTree(
                labelColumnName: "Label", featureColumnName: "Features",
                numberOfTrees: _trees, numberOfLeaves: _leaves,
                learningRate: _lr, minimumExampleCountPerLeaf: _minLeaf);
            _model = pipe.Fit(dv);
        }
        public float[] Predict(float[][] X)
        {
            var dv = _adapter.ToDataView(new Dataset { X = X, y = new float[X.Length] });
            return _ml.Data.CreateEnumerable<MLNetAdapter.ModelOutput>(_model.Transform(dv), false).Select(o => o.Score).ToArray();
        }
        public string SaveState() { using var ms = new MemoryStream(); _ml.Model.Save(_model, _schema, ms); return Convert.ToBase64String(ms.ToArray()); }
        public void LoadState(string s) { using var ms = new MemoryStream(Convert.FromBase64String(s)); _model = _ml.Model.Load(ms, out _schema); }
        public void Dispose() { }
    }

    // ═══════════════ LIGHTGBM ═══════════════
    public class LightGBMModel : IRegressionModel
    {
        private readonly MLContext _ml = new MLContext(seed: 42);
        private readonly MLNetAdapter _adapter;
        private ITransformer _model;
        private DataViewSchema _schema;
        private int _iter = 200, _leaves = 31, _minLeaf = 10;
        private double _lr = 0.1;
        public LightGBMModel() { _adapter = new MLNetAdapter(_ml); }
        public string Name => "LightGBM";
        public Dictionary<string, double> GetDefaultHyperparameters() => new Dictionary<string, double>
        { { "numberOfIterations", 200 }, { "numberOfLeaves", 31 }, { "learningRate", 0.1 }, { "minExamplesPerLeaf", 10 } };
        public void SetHyperparameters(Dictionary<string, double> p)
        {
            if (p.ContainsKey("numberOfIterations")) _iter = Math.Max(10, (int)Math.Round(p["numberOfIterations"]));
            if (p.ContainsKey("numberOfLeaves")) _leaves = Math.Max(2, (int)Math.Round(p["numberOfLeaves"]));
            if (p.ContainsKey("learningRate")) _lr = Math.Max(0.001, Math.Min(1.0, p["learningRate"]));
            if (p.ContainsKey("minExamplesPerLeaf")) _minLeaf = Math.Max(1, (int)Math.Round(p["minExamplesPerLeaf"]));
        }
        public void Train(Dataset data)
        {
            var dv = _adapter.ToDataView(data);
            _schema = dv.Schema;
            var pipe = _ml.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options
            {
                LabelColumnName = "Label",
                FeatureColumnName = "Features",
                NumberOfIterations = _iter,
                NumberOfLeaves = _leaves,
                LearningRate = _lr,
                MinimumExampleCountPerLeaf = _minLeaf
            });
            _model = pipe.Fit(dv);
        }
        public float[] Predict(float[][] X)
        {
            var dv = _adapter.ToDataView(new Dataset { X = X, y = new float[X.Length] });
            return _ml.Data.CreateEnumerable<MLNetAdapter.ModelOutput>(_model.Transform(dv), false).Select(o => o.Score).ToArray();
        }
        public string SaveState() { using var ms = new MemoryStream(); _ml.Model.Save(_model, _schema, ms); return Convert.ToBase64String(ms.ToArray()); }
        public void LoadState(string s) { using var ms = new MemoryStream(Convert.FromBase64String(s)); _model = _ml.Model.Load(ms, out _schema); }
        public void Dispose() { }
    }

    // ═══════════════ ONLINE GRADIENT DESCENT ═══════════════
    public class OgdModel : IRegressionModel
    {
        private readonly MLContext _ml = new MLContext(seed: 42);
        private readonly MLNetAdapter _adapter;
        private ITransformer _model;
        private DataViewSchema _schema;
        private int _iterations = 50;
        private double _lr = 0.01;
        public OgdModel() { _adapter = new MLNetAdapter(_ml); }
        public string Name => "OGD";
        public Dictionary<string, double> GetDefaultHyperparameters() => new Dictionary<string, double>
        { { "iterations", 50 }, { "learningRate", 0.01 } };
        public void SetHyperparameters(Dictionary<string, double> p)
        {
            if (p.ContainsKey("iterations")) _iterations = Math.Max(1, (int)Math.Round(p["iterations"]));
            if (p.ContainsKey("learningRate")) _lr = Math.Max(1e-4, p["learningRate"]);
        }
        public void Train(Dataset data)
        {
            var dv = _adapter.ToDataView(data);
            _schema = dv.Schema;
            var pipe = _ml.Transforms.NormalizeMeanVariance("Features")
                .Append(_ml.Regression.Trainers.OnlineGradientDescent(new OnlineGradientDescentTrainer.Options
                {
                    LearningRate = (float)_lr,
                    NumberOfIterations = _iterations,
                    L2Regularization = 0.001f,
                    LabelColumnName = "Label",
                    FeatureColumnName = "Features"
                }));
            _model = pipe.Fit(dv);
        }
        public float[] Predict(float[][] X)
        {
            var dv = _adapter.ToDataView(new Dataset { X = X, y = new float[X.Length] });
            return _ml.Data.CreateEnumerable<MLNetAdapter.ModelOutput>(_model.Transform(dv), false).Select(o => o.Score).ToArray();
        }
        public string SaveState() { using var ms = new MemoryStream(); _ml.Model.Save(_model, _schema, ms); return Convert.ToBase64String(ms.ToArray()); }
        public void LoadState(string s) { using var ms = new MemoryStream(Convert.FromBase64String(s)); _model = _ml.Model.Load(ms, out _schema); }
        public void Dispose() { }
    }
}