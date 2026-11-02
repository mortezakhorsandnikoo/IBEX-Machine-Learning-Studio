using ArchiAI.ML;
using System;
using System.Data;
using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace ArchiAI.Views
{
    public sealed class CrossValidationTable : Border
    {
        public CrossValidationReport Report { get; private set; }
        private readonly DataGrid grid;
        private readonly TextBlock caption;
        public CrossValidationTable()
        {
            Visibility = Visibility.Collapsed;
            Background = Brushes.White; CornerRadius = new CornerRadius(10);
            Padding = new Thickness(16); Margin = new Thickness(0, 0, 0, 12);
            var panel = new StackPanel(); Child = panel;
            panel.Children.Add(new TextBlock { Text = "K-fold validation results", FontSize = 16,
                FontWeight = FontWeights.SemiBold, Foreground = new SolidColorBrush(Color.FromRgb(23,43,69)) });
            caption = new TextBlock { TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 8, 0, 12),
                Foreground = Brushes.SlateGray, FontSize = 12 };
            panel.Children.Add(caption);
            grid = new DataGrid { IsReadOnly = true, AutoGenerateColumns = true, CanUserAddRows = false,
                CanUserDeleteRows = false, CanUserSortColumns = false, HeadersVisibility = DataGridHeadersVisibility.Column,
                Background = Brushes.White, Foreground = new SolidColorBrush(Color.FromRgb(23,43,69)),
                RowBackground = Brushes.White, AlternatingRowBackground = new SolidColorBrush(Color.FromRgb(241,245,249)),
                BorderThickness = new Thickness(0), GridLinesVisibility = DataGridGridLinesVisibility.Horizontal,
                HorizontalGridLinesBrush = Brushes.LightGray, MaxHeight = 390, MinRowHeight = 30,
                ColumnWidth = new DataGridLength(1, DataGridLengthUnitType.Star), MinColumnWidth = 85 };
            var header = new Style(typeof(System.Windows.Controls.Primitives.DataGridColumnHeader));
            header.Setters.Add(new Setter(Control.BackgroundProperty, new SolidColorBrush(Color.FromRgb(229,243,240))));
            header.Setters.Add(new Setter(Control.ForegroundProperty, Brushes.DarkSlateGray));
            header.Setters.Add(new Setter(Control.PaddingProperty, new Thickness(8)));
            grid.ColumnHeaderStyle = header;
            grid.LoadingRow += (_, e) => e.Row.FontWeight = e.Row.Item is DataRowView row && (string)row[0] == "Combined"
                ? FontWeights.Bold : FontWeights.Normal;
            panel.Children.Add(grid);
        }

        public void Show(CrossValidationReport report)
        {
            Report = report;
            Visibility = report == null ? Visibility.Collapsed : Visibility.Visible;
            grid.ItemsSource = null;
            if (report == null) return;
            string metric = report.Classification ? "accuracy (%)" : "R²";
            var table = new DataTable();
            foreach (string name in new[] { "Fold", "Train rows", "Test rows", "Train " + metric, "Test " + metric }) table.Columns.Add(name);
            string Score(double value) => double.IsFinite(value) ? value.ToString(report.Classification ? "F2" : "F4", CultureInfo.CurrentCulture) : "N/A";
            foreach (var row in report.Rows) table.Rows.Add(row.Fold, row.TrainRows, row.TestRows, Score(row.TrainScore), Score(row.TestScore));
            grid.ItemsSource = table.DefaultView;
            caption.Text = "Combined test: all held-out predictions, with each row tested once. Combined train: all training predictions, including repeated rows across folds. "
                + (report.Classification ? "Accuracy is weighted by row count." : "R² is recalculated from pooled predictions; N/A means fewer than two rows or a constant target. R² can be negative.");
        }
    }
}
