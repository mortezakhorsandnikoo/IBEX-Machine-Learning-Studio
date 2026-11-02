using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Microsoft.Win32;
using ArchiAI.ML;

// NOTE: match your MLView namespace.
namespace ArchiAI.Views
{
    // Manual prediction (use the trained model on new raw inputs) + Save/Load, both algorithms.
    public partial class MLView
    {
        private readonly List<TextBox> _linInputs = new List<TextBox>();
        private readonly List<TextBox> _polyInputs = new List<TextBox>();

        // The active model for each tab (freshly trained or loaded from file).
        private SavedModel _linModel;
        private SavedModel _polyModel;

        // Normalization recipe captured when the Preprocessing "Normalize" step runs.
        // Set by Normalize_Click; cleared in ResetAllTabs / on new CSV load.
        private NormalizationRecipe _normRecipe;

        // ---------------- LINEAR ----------------
        // Call at the end of Train_Click (after DisplayResults()).
        public void BuildLinPredictInputs(PredictionRecipe pipeline = null)
        {
            if (_dataset == null || LinPredictInputs == null) return;
            int target = CmbTarget.SelectedIndex;
            _linModel = new SavedModel
            {
                Type = "Linear",
                Pipeline = pipeline ?? CapturePredictionRecipe(target),
                TargetName = pipeline?.TargetName ?? TargetLabel(target),
                FeatureNames = pipeline?.ModelFeatures ?? FeatureNamesExcept(target).ToArray(),
                Degree = 1,
                Bias = _result.Bias,
                Weights = _result.Weights,
                Means = _result.FeatureMeans,   // null now (Linear normalize is off)
                Stds = _result.FeatureStds,
                Norm = _normRecipe               // scale raw inputs / un-scale output
            };
            BuildInputs(LinPredictInputs, _linInputs, _linModel.InputNames);
            if (LinPredictResult != null) LinPredictResult.Text = "";
        }

        private void LinPredict_Click(object sender, RoutedEventArgs e)
        {
            if (_linModel == null) { LinPredictResult.Text = "Train or load a model first."; return; }
            if (!ReadInputs(_linInputs, out double[] x)) { LinPredictResult.Text = "Enter valid numbers in every box."; return; }
            LinPredictResult.Text = $"{_linModel.TargetName} = {_linModel.PredictRaw(x):G6}";
        }

        private void SaveLinModel_Click(object sender, RoutedEventArgs e)
        {
            if (_linModel == null) { LinPredictResult.Text = "Train a model first."; return; }
            SaveModel(_linModel, "linear_model");
        }

        private void LoadLinModel_Click(object sender, RoutedEventArgs e)
        {
            var m = LoadModel();
            if (m == null) return;
            _linModel = m;
            LinCvTable.Show(null);
            BuildInputs(LinPredictInputs, _linInputs, m.InputNames);
            LinPredictResult.Text = $"Loaded model for {m.TargetName}. Enter values and Predict.";
        }

        // ---------------- POLYNOMIAL ----------------
        public void BuildPolyPredictInputs(PredictionRecipe pipeline = null)
        {
            if (_dataset == null || PolyPredictInputs == null) return;
            int target = PolyTargetCombo.SelectedIndex;
            _polyModel = new SavedModel
            {
                Type = "Polynomial",
                Pipeline = pipeline ?? CapturePredictionRecipe(target),
                TargetName = _polyResult.TargetName,
                FeatureNames = pipeline?.ModelFeatures ?? FeatureNamesExcept(target).ToArray(),
                Degree = _polyResult.Degree,
                Bias = _polyResult.Bias,
                Weights = _polyResult.Weights,
                Means = null,
                Stds = null,
                Norm = _normRecipe
            };
            BuildInputs(PolyPredictInputs, _polyInputs, _polyModel.InputNames);
            if (PolyPredictResult != null) PolyPredictResult.Text = "";
        }

        private void PolyPredict_Click(object sender, RoutedEventArgs e)
        {
            if (_polyModel == null) { PolyPredictResult.Text = "Train or load a model first."; return; }
            if (!ReadInputs(_polyInputs, out double[] x)) { PolyPredictResult.Text = "Enter valid numbers in every box."; return; }
            PolyPredictResult.Text = $"{_polyModel.TargetName} = {_polyModel.PredictRaw(x):G6}";
        }

        private void SavePolyModel_Click(object sender, RoutedEventArgs e)
        {
            if (_polyModel == null) { PolyPredictResult.Text = "Train a model first."; return; }
            SaveModel(_polyModel, "polynomial_model");
        }

        private void LoadPolyModel_Click(object sender, RoutedEventArgs e)
        {
            var m = LoadModel();
            if (m == null) return;
            _polyModel = m;
            PolyCvTable.Show(null);
            BuildInputs(PolyPredictInputs, _polyInputs, m.InputNames);
            PolyPredictResult.Text = $"Loaded model for {m.TargetName}. Enter values and Predict.";
        }

        // ---------------- shared helpers ----------------
        private List<string> FeatureNamesExcept(int targetCol)
        {
            var names = new List<string>();
            for (int c = 0; c < _dataset.ColCount; c++)
                if (c != targetCol) names.Add(_dataset.Headers[c]);
            return names;
        }

        private string TargetLabel(int col) =>
            (_dataset != null && col >= 0 && col < _dataset.ColCount) ? _dataset.Headers[col] : "Prediction";

        private static void SaveModel(SavedModel m, string defaultName)
        {
            var dlg = new SaveFileDialog { Filter = "ArchiAI Model|*.aimodel|Text|*.txt", FileName = defaultName + ".aimodel" };
            if (dlg.ShowDialog() == true) ModelIO.Save(m, dlg.FileName);
        }

        private static SavedModel LoadModel()
        {
            var dlg = new OpenFileDialog { Filter = "ArchiAI Model|*.aimodel|Text|*.txt|All|*.*" };
            return dlg.ShowDialog() == true ? ModelIO.Load(dlg.FileName) : null;
        }

        private static void BuildInputs(StackPanel panel, List<TextBox> store, IEnumerable<string> names)
        {
            store.Clear();
            panel.Children.Clear();
            foreach (var n in names)
            {
                panel.Children.Add(MakeInputRow(n, out var tb));
                store.Add(tb);
            }
        }

        private static UIElement MakeInputRow(string label, out TextBox tb)
        {
            var panel = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 2, 0, 2) };
            panel.Children.Add(new TextBlock
            {
                Text = label,
                Width = 130,
                Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#334155")),
                VerticalAlignment = VerticalAlignment.Center,
                FontSize = 12
            });
            tb = new TextBox
            {
                Width = 110,
                FontSize = 12,
                Background = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#F1F5F9")),
                Foreground = Brushes.DarkSlateGray,
                BorderBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#E2E7EF")),
                Padding = new Thickness(6, 3, 6, 3)
            };
            panel.Children.Add(tb);
            return panel;
        }

        private static bool ReadInputs(List<TextBox> inputs, out double[] x)
        {
            x = new double[inputs.Count];
            if (inputs.Count == 0) return false;
            for (int i = 0; i < inputs.Count; i++)
            {
                if (!double.TryParse(inputs[i].Text?.Trim(), NumberStyles.Any,
                        CultureInfo.InvariantCulture, out x[i]) || !double.IsFinite(x[i]))
                    return false;
            }
            return true;
        }
    }
}
