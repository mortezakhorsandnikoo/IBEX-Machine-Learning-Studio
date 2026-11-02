using System;
using System.Windows;
using System.Windows.Controls;
namespace ArchiAI.Views
{
    public partial class WelcomeView : Page
    {
        public event Action<int> OpenSection;
        public WelcomeView() => InitializeComponent();
        private void Support_RequestNavigate(object sender, System.Windows.Navigation.RequestNavigateEventArgs e)
        {
            e.Handled = true;
            try
            {
                System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(e.Uri.AbsoluteUri) { UseShellExecute = true });
            }
            catch (Exception)
            {
                MessageBox.Show("Could not open your browser. Visit https://ko-fi.com/mortezakhorsand to support IBEX.", "IBEX");
            }
        }
        private void Task_Click(object sender, RoutedEventArgs e) => OpenSection?.Invoke(int.Parse((string)((Button)sender).Tag));
    }
}
