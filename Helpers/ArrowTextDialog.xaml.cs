using System.Windows;

namespace ArchiAI.Helpers
{
    public partial class ArrowTextDialog : Window
    {
        public string LabelText { get; private set; } = "";

        public ArrowTextDialog(string defaultText = "")
        {
            InitializeComponent();
            TxtLabel.Text = defaultText;
            TxtLabel.SelectAll();
            TxtLabel.Focus();
        }

        private void Ok_Click(object sender, RoutedEventArgs e)
        {
            LabelText = TxtLabel.Text.Trim();
            DialogResult = true;
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
        }
    }
}