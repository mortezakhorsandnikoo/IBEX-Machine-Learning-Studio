using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

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
            PreviewKeyDown += ZoomKeys;
            PreviewMouseWheel += ZoomWheel;
        }

        // ---- Page zoom: Ctrl + mouse wheel, Ctrl + plus/minus, Ctrl + 0 ----
        // Scales every page (Home, workspace pages, Help) like browser zoom; the layout reflows
        // so panels keep fitting the window and scroll where needed.
        private const double MinZoom = 0.5, MaxZoom = 2.5, ZoomStep = 0.1;

        public double Zoom => PageZoom.ScaleX;

        public void SetZoom(double zoom)
        {
            zoom = Math.Round(Math.Clamp(zoom, MinZoom, MaxZoom), 2);
            PageZoom.ScaleX = PageZoom.ScaleY = zoom;
            ZoomResetButton.Content = $"{zoom * 100:0}%";
        }

        private void ZoomWheel(object sender, MouseWheelEventArgs e)
        {
            if ((Keyboard.Modifiers & ModifierKeys.Control) == 0) return;
            SetZoom(Zoom + Math.Sign(e.Delta) * ZoomStep);
            e.Handled = true; // keep charts and scroll views from also reacting
        }

        private void ZoomKeys(object sender, KeyEventArgs e)
        {
            if ((Keyboard.Modifiers & ModifierKeys.Control) == 0) return;
            switch (e.Key)
            {
                case Key.OemPlus: case Key.Add: SetZoom(Zoom + ZoomStep); break;
                case Key.OemMinus: case Key.Subtract: SetZoom(Zoom - ZoomStep); break;
                case Key.D0: case Key.NumPad0: SetZoom(1); break;
                default: return;
            }
            e.Handled = true;
        }

        private void ZoomReset_Click(object sender, RoutedEventArgs e) => SetZoom(1);

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
