using ArchiAI.ML;
using System.Linq;
using System.Windows.Controls;

namespace ArchiAI.Views
{
    public partial class MLView
    {
        // Algorithms train on the prepared _dataset (normalized features). The Dataset page and
        // its data preview show the same rows with every normalization step undone, so users
        // always read their original values.
        private MLDataset DisplayDataset()
        {
            if (_dataset == null) return null;
            var recipes = _preprocessingSteps.Where(s => s.Normalization != null)
                .Select(s => s.Normalization).Reverse().ToArray();
            if (recipes.Length == 0) return _dataset;
            var headers = _dataset.Headers;
            var rows = _dataset.Rows.Select(row =>
            {
                var values = (double[])row.Clone();
                foreach (var recipe in recipes)
                    for (int c = 0; c < values.Length; c++)
                        values[c] = recipe.Inverse(headers[c], values[c]);
                return values;
            }).ToArray();
            return MLDataset.FromRows(headers, rows, _dataset.RawRows);
        }

        private bool HasNormalizationStep => _preprocessingSteps.Any(s => s.Normalization != null);

        // Target column offered to Normalize: defaults to the last column, like the modeling pages.
        private void FillNormTargetCombo()
        {
            if (CmbNormTarget == null) return;
            CmbNormTarget.Items.Clear();
            if (_dataset == null) return;
            foreach (var h in _dataset.Headers) CmbNormTarget.Items.Add(h);
            CmbNormTarget.Items.Add(new ComboBoxItem { Content = "No target (normalize all columns)" });
            CmbNormTarget.SelectedIndex = _dataset.ColCount - 1;
        }

        // Index of the column kept in original units, or -1 when every column is scaled.
        private int NormTargetIndex()
        {
            int i = CmbNormTarget?.SelectedIndex ?? -1;
            return _dataset != null && i >= 0 && i < _dataset.ColCount ? i : -1;
        }
    }
}
