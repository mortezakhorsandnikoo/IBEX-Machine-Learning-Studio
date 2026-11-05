using System.Reflection;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using ArchiAI;
using ArchiAI.Help;
using ArchiAI.ML;
using ArchiAI.Views;

internal static partial class Smoke
{
    private static async Task CheckHelp(MainWindow window)
    {
        var catalog = HelpCatalog.Load();
        Assert(catalog.Count == 51, "All 51 help topics load from embedded resources");
        var pdfBytes = HelpDownloads.ReadCheatSheet();
        Assert(System.Text.Encoding.ASCII.GetString(pdfBytes, 0, 5) == "%PDF-", "Cheat sheet loads as a PDF from the embedded resource");
        var pdfPath = System.IO.Path.Combine(output, HelpDownloads.CheatSheetFileName);
        HelpDownloads.SaveCheatSheet(pdfPath);
        Assert(System.IO.File.ReadAllBytes(pdfPath).SequenceEqual(pdfBytes), "Saved cheat sheet exactly matches the embedded PDF");
        var viewPath = HelpDownloads.GetCheatSheetForViewing();
        Assert(System.IO.File.ReadAllBytes(viewPath).SequenceEqual(pdfBytes), "Open/print extracts the same PDF without external files");
        Assert(HelpDownloads.GetCheatSheetForViewing() == viewPath, "Viewing reuses the content-versioned local PDF");
        Assert(HelpCatalog.Search(catalog, "print cheat sheet").Any(m => m.Topic.Id == "cheat-sheet"), "Printable cheat sheet is discoverable through help search");
        foreach (var topic in catalog)
        {
            Assert(topic.Related.All(id => catalog.Any(t => t.Id == id)), "Valid related links: " + topic.Id);
            foreach (var b in topic.Blocks.Where(b => b.Kind == "image"))
            {
                using var stream = Application.GetResourceStream(HelpCatalog.ResourceUri("Images/" + b.Image)).Stream;
                Assert(stream.Length > 0, "Embedded illustration: " + b.Image);
            }
        }
        Assert(HelpCatalog.Search(catalog, null).Count == catalog.Count, "Empty query returns contents");
        Assert(HelpCatalog.Search(catalog, "PCA original prediction").Any(m => m.Topic.Id == "pca"), "Multiple words find original-feature PCA instructions");
        Assert(HelpCatalog.Search(catalog, "K-fold combined").Any(m => m.Topic.Id == "cross-validation"), "Punctuation and case are normalized for search");
        Assert(HelpCatalog.Search(catalog, "epsilon").Any(m => m.Topic.Id == "rbf"), "Search covers parameter names in the full article");
        Assert(HelpCatalog.Search(catalog, "", "Clustering algorithms").Count == 4, "Category filtering isolates four clustering algorithms");
        Assert(HelpCatalog.Search(catalog, "zzzznoresult").Count == 0, "Unknown term returns no results");
        Assert(catalog.Count(t => t.Category == "Regression algorithms") == 11 && catalog.Count(t => t.Category == "Classification algorithms") == 9, "All regression and classification algorithms documented");

        var frame = (Frame)window.FindName("MainFrame");
        window.ShowSection(2);
        await Settle(window);
        var workspace = (MLView)frame.Content;
        string sample = System.IO.Path.Combine(output, "synthetic-clusters.csv");
        var rows = Enumerable.Range(0, 90).Select(i => new[] { (i / 30) * 5.0 + Math.Sin(i * 1.7), (i / 30 % 2) * 5.0 + Math.Cos(i * 1.3) });
        System.IO.File.WriteAllText(sample, "Feature A,Feature B\n" + string.Join("\n", rows.Select(r => string.Join(",", r.Select(v => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture))))));
        Assert(workspace.LoadDataset(sample), "Synthetic illustration dataset loads");
        window.ShowSection(4); await Settle(window);
        Click(workspace, "ClustRun_Click");
        await Until(() => Field<ClusterResult>(workspace, "_clustResult") != null, "Synthetic clustering completes for illustration");
        var trained = Field<ClusterResult>(workspace, "_clustResult");
        await Settle(window); Render(window, "clustering");
        window.ShowSection(2);
        window.OpenHelp();
        await Settle(window);
        var help = (HelpView)frame.Content;
        Assert(help.CurrentTopicId == "regression", "Context help opens current workspace instructions");
        foreach (var t in catalog)
        {
            help.NavigateTo(t.Id);
            Assert(help.CurrentTopicId == t.Id, "Article renders: " + t.Id);
        }
        help.NavigateTo("start");
        await Settle(window); Render(window, "help-home");
        var startArticle = (FlowDocumentScrollViewer)help.FindName("ArticleViewer");
        var illustration = (Button)startArticle.Document.Blocks.OfType<BlockUIContainer>().First().Child;
        illustration.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
        var zoomWindow = Application.Current.Windows.Cast<Window>().Single(w => w.Title == "Guide illustration");
        Assert(zoomWindow.IsVisible && zoomWindow.Owner == window, "Illustrations open in an owned offline zoom window");
        zoomWindow.Close();
        ((TextBox)help.FindName("SearchBox")).Text = "PCA prediction";
        await Settle(window); Render(window, "help-search");
        Assert(((ListBox)help.FindName("TopicsList")).Items.Count > 0, "Live search updates topic list");
        Assert(help.CurrentTopicId == "pca", "Search opens the highest-ranked matching article");
        var article = (FlowDocumentScrollViewer)help.FindName("ArticleViewer");
        Assert(new TextRange(article.Document.ContentStart, article.Document.ContentEnd).Text.Contains("PCA"), "Search displays a matching article");
        ((TextBox)help.FindName("SearchBox")).Text = "zzzznoresult";
        Assert(help.CurrentTopicId == null && new TextRange(article.Document.ContentStart, article.Document.ContentEnd).Text.Contains("No matching topics"), "No-results state clears the previous article");
        help.NavigateTo("pca");
        window.Width = 1120; window.Height = 700;
        await Settle(window); Render(window, "help-small");
        var sidebarCheat = (Button)window.FindName("CheatSheetButton");
        var sidebarHelp = (RadioButton)window.FindName("HelpNav");
        var cheatBounds = sidebarCheat.TransformToAncestor(window).TransformBounds(new Rect(sidebarCheat.RenderSize));
        var helpBounds = sidebarHelp.TransformToAncestor(window).TransformBounds(new Rect(sidebarHelp.RenderSize));
        Assert(cheatBounds.Top >= helpBounds.Bottom && cheatBounds.Bottom < window.ActualHeight && cheatBounds.Left >= 0,
            "Direct cheat-sheet button stays on screen below Help at minimum window size");
        Assert(((Button)help.FindName("SaveCheatSheetButton")).IsVisible && ((Button)help.FindName("OpenCheatSheetButton")).IsVisible, "Cheat sheet save and open/print actions are visible at minimum size");
        Assert(article.ActualWidth > 500 && article.ActualHeight > 350, "Help remains readable at minimum window size");
        help.NavigateTo("start");
        await Settle(window); Render(window, "help-small-image");
        typeof(HelpView).GetMethod("Back_Click", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(help, new object[] { help, new RoutedEventArgs() });
        await Settle(window);
        Assert(ReferenceEquals(frame.Content, workspace), "Back returns the same workspace instance");
        Assert(ReferenceEquals(trained, Field<ClusterResult>(workspace, "_clustResult")), "Opening and searching help preserves trained models");
        for (int i = 0; i < 7; i++)
        {
            window.ShowSection(i); window.OpenHelp();
            Assert(help.CurrentTopicId == new[] { "dataset", "prepare", "regression", "classification", "clustering", "compare", "visualize" }[i], "Context help for page " + i);
        }
        window.ShowSection(-1); window.OpenHelp();
        Assert(help.CurrentTopicId == "start", "Home help opens getting started");
    }
}
