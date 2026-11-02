using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace ArchiAI.Helpers
{
    public partial class ColorPickerDialog : Window
    {
        public Color SelectedColor { get; private set; }

        public ColorPickerDialog(Color c)
        {
            InitializeComponent();
            SR.Value = c.R;
            SG.Value = c.G;
            SB.Value = c.B;
            Update();
        }

        private void Update()
        {
            byte r = (byte)SR.Value, g = (byte)SG.Value, b = (byte)SB.Value;
            SelectedColor = Color.FromRgb(r, g, b);
            if (Prev != null) Prev.Background = new SolidColorBrush(SelectedColor);
            if (LR != null) LR.Text = r.ToString();
            if (LG != null) LG.Text = g.ToString();
            if (LB != null) LB.Text = b.ToString();
        }

        private void S_Changed(object sender, RoutedPropertyChangedEventArgs<double> e) => Update();

        private void P(object sender, RoutedEventArgs e)
        {
            if (sender is Button btn && btn.Tag is string t)
            {
                var p = t.Split(',');
                if (p.Length == 3)
                {
                    SR.Value = int.Parse(p[0]);
                    SG.Value = int.Parse(p[1]);
                    SB.Value = int.Parse(p[2]);
                }
            }
        }

        private void Ok(object sender, RoutedEventArgs e)
        {
            Update();
            DialogResult = true;
        }

        private void Cancel(object sender, RoutedEventArgs e) => DialogResult = false;
    }
}