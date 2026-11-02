using System.Globalization;
using System.Windows;

namespace ArchiAI.Helpers
{
    public partial class InputDialog : Window
    {
        public double ResultValue { get; private set; }

        public InputDialog(string title, string prompt, double defaultValue = 1.0)
        {
            InitializeComponent();
            Title = title;
            TxtPrompt.Text = prompt;
            TxtValue.Text = defaultValue.ToString(CultureInfo.InvariantCulture);
            TxtValue.SelectAll();
            TxtValue.Focus();
        }

        private void Ok_Click(object sender, RoutedEventArgs e)
        {
            if (double.TryParse(TxtValue.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out double val) && val > 0)
            {
                ResultValue = val;
                DialogResult = true;
            }
            else
            {
                MessageBox.Show("Please enter a valid positive number.", "Invalid Input");
            }
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
        }
    }
}