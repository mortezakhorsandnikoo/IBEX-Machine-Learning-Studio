using ArchiAI.Help;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ArchiAI.Views
{
    public partial class HelpView : Page
    {
        public event Action ReturnToWorkspace;
        public IReadOnlyList<HelpTopic> Topics { get; }
        public string CurrentTopicId { get; private set; }
        private bool updating;
        private static SolidColorBrush Brush(string hex) => new SolidColorBrush((Color)ColorConverter.ConvertFromString(hex));
        public HelpView()
        {
            InitializeComponent();
            Topics = HelpCatalog.Load();
            CategoryBox.ItemsSource = new[] { "All topics" }.Concat(Topics.Select(t => t.Category).Distinct()).ToArray();
            CategoryBox.SelectedIndex = 0;
            Refresh();
            PreviewKeyDown += (_, e) => { if (e.Key == Key.F && Keyboard.Modifiers == ModifierKeys.Control) { SearchBox.Focus(); SearchBox.SelectAll(); e.Handled = true; } };
        }
        public void NavigateTo(string id)
        {
            updating = true; SearchBox.Text = ""; CategoryBox.SelectedIndex = 0; updating = false;
            Refresh(id);
        }
        private void Search_Changed(object sender, TextChangedEventArgs e) { if (Topics != null && !updating) Refresh(preserveSelection: false); }
        private void Category_Changed(object sender, SelectionChangedEventArgs e) { if (Topics != null && !updating) Refresh(preserveSelection: false); }
        private void Clear_Click(object sender, RoutedEventArgs e) { NavigateTo("start"); SearchBox.Focus(); }
        private void Back_Click(object sender, RoutedEventArgs e) => ReturnToWorkspace?.Invoke();
        private void SaveCheatSheet_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Microsoft.Win32.SaveFileDialog { Title = "Save machine-learning cheat sheet", FileName = HelpDownloads.CheatSheetFileName, Filter = "PDF document (*.pdf)|*.pdf", DefaultExt = ".pdf", AddExtension = true, OverwritePrompt = true };
            if (dialog.ShowDialog(Window.GetWindow(this)) != true) return;
            try { HelpDownloads.SaveCheatSheet(dialog.FileName); }
            catch (Exception ex) { MessageBox.Show(Window.GetWindow(this), "The PDF could not be saved. Choose a writable folder and try again.\n\n" + ex.Message, "Save cheat sheet", MessageBoxButton.OK, MessageBoxImage.Error); }
        }
        private void OpenCheatSheet_Click(object sender, RoutedEventArgs e)
        {
            HelpDownloads.OpenCheatSheet(Window.GetWindow(this));
        }
        private void Refresh(string selectId = null, bool preserveSelection = true)
        {
            string previous = selectId ?? (preserveSelection ? CurrentTopicId : null);
            var matches = HelpCatalog.Search(Topics, SearchBox.Text, CategoryBox.SelectedIndex <= 0 ? null : CategoryBox.SelectedItem as string);
            TopicsList.ItemsSource = matches;
            ResultsStatus.Text = matches.Count == 0 ? "No matches. Try a shorter term or choose All topics." : $"{matches.Count} topic{(matches.Count == 1 ? "" : "s")}";
            TopicsList.SelectedItem = matches.FirstOrDefault(m => m.Topic.Id == previous) ?? matches.FirstOrDefault();
            if (TopicsList.SelectedItem != null) TopicsList.ScrollIntoView(TopicsList.SelectedItem);
            if (matches.Count == 0)
            {
                CurrentTopicId = null;
                ArticleViewer.Document = Document("No matching topics", "Try words such as CSV, PCA, accuracy, K-fold, normalization or save model. Search looks through the full guide, including parameter names and troubleshooting.");
            }
        }
        private FlowDocument Document(string title, string summary)
        {
            var doc = new FlowDocument { FontFamily = new FontFamily("Segoe UI"), FontSize = 14, TextAlignment = TextAlignment.Left, Foreground = Brush("#172B45"), Background = Brushes.White, PagePadding = new Thickness(24), ColumnWidth = 1000 };
            doc.Blocks.Add(new Paragraph(new Run(title)) { FontSize = 26, FontWeight = FontWeights.SemiBold, Margin = new Thickness(0, 0, 0, 12) });
            doc.Blocks.Add(new Paragraph(new Run(summary)) { Foreground = Brush("#64748B"), Margin = new Thickness(0, 0, 0, 18) });
            return doc;
        }
        private Paragraph Text(string text)
        {
            var p = new Paragraph { Margin = new Thickness(0, 0, 0, 12), LineHeight = 22 };
            var terms = HelpCatalog.Tokens(SearchBox.Text);
            if (terms.Length == 0) { p.Inlines.Add(new Run(text ?? "")); return p; }
            var regex = new Regex(string.Join("|", terms.OrderByDescending(t => t.Length).Select(Regex.Escape)), RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
            int position = 0;
            foreach (Match match in regex.Matches(text ?? ""))
            {
                p.Inlines.Add(new Run(text.Substring(position, match.Index - position)));
                p.Inlines.Add(new Run(match.Value) { Background = Brush("#FFF0B8"), Foreground = Brush("#172B45") });
                position = match.Index + match.Length;
            }
            p.Inlines.Add(new Run((text ?? "").Substring(position))); return p;
        }
        private void Topic_Changed(object sender, SelectionChangedEventArgs e)
        {
            if (TopicsList.SelectedItem is not HelpMatch match) return;
            var topic = match.Topic; CurrentTopicId = topic.Id;
            var doc = Document(topic.Title, topic.Summary);
            foreach (var b in topic.Blocks)
            {
                if (b.Kind == "image")
                {
                    var bitmap = new BitmapImage(HelpCatalog.ResourceUri("Images/" + b.Image));
                    var image = new Image { Source = bitmap, Stretch = Stretch.Uniform, MaxWidth = 920, MaxHeight = 600, HorizontalAlignment = HorizontalAlignment.Left };
                    System.Windows.Automation.AutomationProperties.SetName(image, b.Caption);
                    var enlarge = new Button { Content = image, Background = Brushes.White, BorderThickness = new Thickness(0), Padding = new Thickness(0), HorizontalContentAlignment = HorizontalAlignment.Stretch, ToolTip = "Open illustration and zoom", Cursor = Cursors.Hand };
                    System.Windows.Automation.AutomationProperties.SetName(enlarge, "Enlarge: " + b.Caption);
                    enlarge.Click += (_, _) => ShowIllustration(bitmap, b.Caption);
                    doc.Blocks.Add(new BlockUIContainer(enlarge) { Margin = new Thickness(0, 8, 0, 6) });
                    doc.Blocks.Add(new Paragraph(new Run(b.Caption)) { FontSize = 12, Foreground = Brush("#64748B"), Margin = new Thickness(0, 0, 0, 18) });
                }
                else if (b.Kind == "steps" || b.Kind == "bullets")
                {
                    var list = new List { MarkerStyle = b.Kind == "steps" ? TextMarkerStyle.Decimal : TextMarkerStyle.Disc, Margin = new Thickness(0, 0, 0, 12), Padding = new Thickness(20, 0, 0, 0) };
                    foreach (var item in b.Items ?? Array.Empty<string>()) list.ListItems.Add(new ListItem(Text(item)));
                    doc.Blocks.Add(list);
                }
                else
                {
                    var p = Text(b.Text);
                    if (b.Kind == "heading") { p.FontSize = 18; p.FontWeight = FontWeights.SemiBold; p.Foreground = Brush("#087F73"); p.Margin = new Thickness(0, 12, 0, 8); }
                    if (b.Kind == "note") { p.Background = Brush("#EDF6F3"); p.Padding = new Thickness(12); }
                    doc.Blocks.Add(p);
                }
            }
            if (topic.Related.Length > 0)
            {
                doc.Blocks.Add(new Paragraph(new Run("Related topics")) { FontWeight = FontWeights.SemiBold, Foreground = Brush("#087F73") });
                foreach (var id in topic.Related)
                {
                    var related = Topics.FirstOrDefault(t => t.Id == id); if (related == null) continue;
                    var link = new Hyperlink(new Run(related.Title)) { Foreground = Brush("#087F73") };
                    link.Click += (_, _) => NavigateTo(id);
                    doc.Blocks.Add(new Paragraph(link));
                }
            }
            ArticleViewer.Document = doc;
            Dispatcher.BeginInvoke(new Action(() => doc.Blocks.FirstBlock?.BringIntoView()));
        }
        private void ShowIllustration(BitmapImage bitmap, string caption)
        {
            var owner = Window.GetWindow(this);
            var popup = new Window { Title = "Guide illustration", Owner = owner, Width = Math.Min(1100, SystemParameters.WorkArea.Width), Height = Math.Min(780, SystemParameters.WorkArea.Height), WindowStartupLocation = WindowStartupLocation.CenterOwner, Background = Brushes.White };
            var panel = new DockPanel { Margin = new Thickness(16) };
            var header = new StackPanel(); DockPanel.SetDock(header, Dock.Top);
            header.Children.Add(new TextBlock { Text = caption, TextWrapping = TextWrapping.Wrap, Foreground = Brush("#172B45"), FontSize = 15 });
            header.Children.Add(new TextBlock { Text = "Zoom to read details. Press Esc to close.", Margin = new Thickness(0, 8, 0, 4), Foreground = Brush("#64748B") });
            var zoom = new Slider { Minimum = 10, Maximum = 200, Value = Math.Min(100, (popup.Width - 70) * 100 / bitmap.PixelWidth), Margin = new Thickness(0, 4, 0, 12), SmallChange = 10, LargeChange = 25 };
            System.Windows.Automation.AutomationProperties.SetName(zoom, "Illustration zoom percentage");
            var enlarged = new Image { Source = bitmap, Stretch = Stretch.Uniform, HorizontalAlignment = HorizontalAlignment.Left, VerticalAlignment = VerticalAlignment.Top };
            void Resize() { enlarged.Width = bitmap.PixelWidth * zoom.Value / 100; }
            zoom.ValueChanged += (_, _) => Resize(); Resize();
            header.Children.Add(zoom); panel.Children.Add(header);
            panel.Children.Add(new ScrollViewer { Content = enlarged, HorizontalScrollBarVisibility = ScrollBarVisibility.Auto, VerticalScrollBarVisibility = ScrollBarVisibility.Auto });
            popup.Content = panel;
            popup.PreviewKeyDown += (_, e) => { if (e.Key == Key.Escape) { popup.Close(); e.Handled = true; } };
            popup.Show();
        }
    }
}
