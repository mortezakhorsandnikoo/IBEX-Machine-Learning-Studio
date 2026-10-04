using System;
using System.Windows;
using System.Windows.Controls;

namespace ArchiAI
{
    public partial class MainWindow : Window
    {
        private Views.MLView _workspace;
        private readonly Views.WelcomeView _welcome;
        private Views.HelpView _help;
        private int _lastWorkspaceSection = -1;
        private static readonly string[] HelpTopics = { "dataset", "prepare", "regression", "classification", "clustering", "compare", "visualize" };
        private static readonly string[] Titles = { "Dataset", "Prepare data", "Regression", "Classification", "Clustering", "Compare models", "Visualize" };
        private static readonly string[] Descriptions = {
            "Import a CSV and get to know your features.",
            "Clean, encode and transform your dataset before training.",
            "Predict continuous values. Inspect fit, error and feature weights.",
            "Learn categories and evaluate classification performance.",
            "Discover structure and groups in your data.",
            "Compare the results of your experiments in this session.",
            "Explore relationships and turn your data into clear figures." };

        public MainWindow()
        {
            InitializeComponent();
            _welcome = new Views.WelcomeView();
            _welcome.OpenSection += ShowSection;
            ShowSection(-1);
            PreviewKeyDown += (_, e) => { if (e.Key == System.Windows.Input.Key.F1) { OpenHelp(); e.Handled = true; } };
        }

        private Views.MLView Workspace => _workspace ??= new Views.MLView();

        public void ShowSection(int index)
        {
            if (index < -1 || index > 7) return;
            if (index != 7) _lastWorkspaceSection = index;
            var nav = new[] { DataNav, PrepNav, RegNav, ClsNav, ClustNav, CompareNav, VizNav };
            if (index == 7)
            {
                if (_help == null)
                {
                    _help = new Views.HelpView();
                    _help.ReturnToWorkspace += () => ShowSection(_lastWorkspaceSection);
                }
                HelpNav.IsChecked = true;
                SectionTitle.Text = "Help and user guide";
                SectionSubtitle.Text = "Find instructions, algorithm settings and illustrated examples. Available offline.";
                MainFrame.Content = _help;
            }
            else if (index < 0)
            {
                HomeNav.IsChecked = true;
                SectionTitle.Text = "Welcome to your studio";
                SectionSubtitle.Text = "A clear path from data to discovery.";
                MainFrame.Content = _welcome;
            }
            else
            {
                nav[index].IsChecked = true;
                SectionTitle.Text = Titles[index];
                SectionSubtitle.Text = Descriptions[index];
                MainFrame.Content = Workspace;
                Workspace.SelectSection(index);
            }
            // The sidebar owns navigation; do not retain a second, hidden navigation history.
            while (MainFrame.CanGoBack) MainFrame.RemoveBackEntry();
        }

        private void Navigate_Click(object sender, RoutedEventArgs e) => ShowSection(int.Parse((string)((RadioButton)sender).Tag));
        private void Help_Click(object sender, RoutedEventArgs e) => OpenHelp();
        private void CheatSheet_Click(object sender, RoutedEventArgs e) => Help.HelpDownloads.OpenCheatSheet(this);
        public void OpenHelp()
        {
            var topic = _lastWorkspaceSection >= 0 ? HelpTopics[_lastWorkspaceSection] : "start";
            ShowSection(7);
            _help.NavigateTo(topic);
        }
        public void OpenSample()
        {
            ShowSection(0);
            var path = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Samples", "curve.csv");
            if (Workspace.LoadDataset(path))
            {
                ShowSection(2);
                Workspace.SelectPolynomial();
            }
        }
    }
}
