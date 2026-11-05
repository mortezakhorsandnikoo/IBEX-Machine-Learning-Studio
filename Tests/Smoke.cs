using System.IO;
using System.Reflection;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using ArchiAI;
using ArchiAI.ML;
using ArchiAI.Views;
using LiveChartsCore.SkiaSharpView.WPF;

internal static partial class Smoke
{
    private static int assertions;
    private static string output;
    [STAThread]
    public static int Main(string[] args)
    {
        output = args.Length > 0 ? Path.GetFullPath(args[0]) : Path.Combine(AppContext.BaseDirectory, "verification");
        Directory.CreateDirectory(output);
        var app = new Application();
        app.Resources.MergedDictionaries.Add(new ResourceDictionary { Source = new Uri("/IBEX.MLStudio;component/Themes/Light.xaml", UriKind.Relative) });
        app.Resources.MergedDictionaries.Add(new ResourceDictionary { Source = new Uri("/IBEX.MLStudio;component/Themes/Modern.xaml", UriKind.Relative) });

        app.ShutdownMode = ShutdownMode.OnExplicitShutdown;
        var window = new MainWindow { WindowStartupLocation = WindowStartupLocation.Manual, Left = -16000, Top = -16000, ShowInTaskbar = false, ShowActivated = false };
        window.Loaded += async (_, _) =>
        {
            try
            {
                if (Environment.GetEnvironmentVariable("IBEX_HELP_TEST") == "1") await CheckHelp(window);
                else await Check(window);
                if (Environment.GetEnvironmentVariable("IBEX_RELEASE_TEST") == "1") CheckReleaseRuntime();
                Console.WriteLine($"PASS: {assertions} checks; renders saved to {output}");
                app.Shutdown(0);
            }
            catch (Exception ex) { Console.Error.WriteLine(ex); app.Shutdown(1); }
        };
        return app.Run(window);
    }
    private static void Assert(bool condition, string text)
    {
        if (!condition) throw new Exception(text);
        assertions++;
        Console.WriteLine("PASS: " + text);
    }
    private static T Find<T>(MLView view, string name) => (T)view.FindName(name);
    private static void Click(MLView view, string handler) => typeof(MLView).GetMethod(handler, BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(view, new object[] { view, new RoutedEventArgs() });
    private static async Task Settle(Window window) { window.UpdateLayout(); await Task.Delay(650); window.UpdateLayout(); }
    private static void Render(Window window, string name)
    {
        var content = (FrameworkElement)window.Content;
        double scale = Environment.GetEnvironmentVariable("IBEX_RENDER_SCALE") == "3" ? 3 : 1;
        var bitmap = new RenderTargetBitmap((int)(content.ActualWidth * scale), (int)(content.ActualHeight * scale), 96 * scale, 96 * scale, PixelFormats.Pbgra32);
        bitmap.Render(content);
        var png = new PngBitmapEncoder();
        png.Frames.Add(BitmapFrame.Create(bitmap));
        using var stream = File.Create(Path.Combine(output, name + ".png"));
        png.Save(stream);
    }
    private static void AssertChartPixels(CartesianChart chart)
    {
        var bitmap = new RenderTargetBitmap((int)chart.ActualWidth, (int)chart.ActualHeight, 96, 96, PixelFormats.Pbgra32);
        bitmap.Render(chart);
        int stride = bitmap.PixelWidth * 4;
        var pixels = new byte[stride * bitmap.PixelHeight];
        bitmap.CopyPixels(pixels, stride, 0);
        int marks = 0;
        for (int i = 0; i < pixels.Length; i += 4)
            if (pixels[i + 3] > 100 && pixels[i + 1] > pixels[i + 2] + 25 && pixels[i + 2] < 100) marks++;
        Assert(marks > 30, chart.Name + " contains plotted teal pixels on first training (" + marks + ")");
    }
    private static async Task Train(MLView view)
    {
        Click(view, "PolyTrain_Click");
        var deadline = DateTime.UtcNow.AddSeconds(30);
        while (view.IsTrainingPolynomial && DateTime.UtcNow < deadline) await Task.Delay(30);
        Assert(!view.IsTrainingPolynomial && view.IsEnabled, "Training returns control to the workspace");
        Assert(Find<Grid>(view, "PolyChartGrid").Visibility == Visibility.Visible, Find<TextBlock>(view, "PolyChartStatus").Text);
    }
    private static async Task Check(MainWindow window)
    {
        await Settle(window);
        var frame = (Frame)window.FindName("MainFrame");
        Assert(window.Title == "IBEX Machine Learning Studio", "IBEX window branding");
        Assert(frame.Content is WelcomeView, "Application starts at the welcome page");
        Render(window, "home");
        window.Width = 1120; window.Height = 700;
        await Settle(window); Render(window, "home-small");
        window.Width = 1440; window.Height = 900;
        window.OpenSample();
        await Settle(window);
        var view = (MLView)frame.Content;
        Assert(Find<ComboBox>(view, "CmbRegAlgo").SelectedIndex == 1, "Sample opens Polynomial regression");
        Assert(Find<ComboBox>(view, "PolyTargetCombo").SelectedItem.ToString() == "Response", "Sample target selected");
        await Train(view);
        await Settle(window);
        foreach (string name in new[] { "PolyCostChart", "PolyScatterChart", "PolyResidualChart", "PolyImportanceChart" })
        {
            var chart = Find<CartesianChart>(view, name);
            Assert(chart.IsVisible && chart.ActualWidth > 200 && chart.ActualHeight >= 200 && chart.Series.Any(), name + " has rendered bounds and series");
        }
        AssertChartPixels(Find<CartesianChart>(view, "PolyCostChart"));
        AssertChartPixels(Find<CartesianChart>(view, "PolyScatterChart"));
        Render(window, "polynomial");
        var result = (PolynomialRegressionResult)typeof(MLView).GetField("_polyResult", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(view)!;
        Assert(result.R2 > 0.98 && result.CostHistory.All(double.IsFinite), "Sample polynomial fit is accurate and finite");
        Assert(Find<StackPanel>(view, "PolyPredictInputs").Children.Count > 0, "Training creates prediction inputs");
        window.ShowSection(-1); await Settle(window);
        window.ShowSection(2); await Settle(window);
        Assert(ReferenceEquals(view, frame.Content) && Find<Grid>(view, "PolyChartGrid").IsVisible, "Home round-trip preserves the trained workspace");
        window.Width = 1120; window.Height = 700;
        await Settle(window); Render(window, "polynomial-small");
        foreach (string name in new[] { "PolyCostChart", "PolyScatterChart", "PolyResidualChart", "PolyImportanceChart" })
            Assert(Find<CartesianChart>(view, name).ActualWidth > 200, name + " retains usable width at minimum window size");
        Find<CheckBox>(view, "PolyUseCV").IsChecked = true;
        await Train(view);
        Assert(Find<TextBlock>(view, "PolyChartStatus").Text.Contains("held-out folds"), "CV explains final all-data charts separately from held-out metrics");
        typeof(MLView).GetMethod("RefreshDataPreview", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(view, null);
        Assert(!Find<CartesianChart>(view, "PolyCostChart").Series.Any(), "Preprocessed data invalidates previous polynomial charts");
        Assert(Find<StackPanel>(view, "PolyPredictInputs").Children.Count == 0, "Preprocessed data invalidates the previous prediction model");
        var sample = Path.Combine(AppContext.BaseDirectory, "Samples", "curve.csv");
        Assert(view.LoadDataset(sample), "Reload sample successfully");
        Assert(!Find<CartesianChart>(view, "PolyScatterChart").Series.Any(), "New dataset clears old polynomial charts");
        Assert(!view.LoadDataset(Path.Combine(output, "missing.csv")), "Invalid import reports failure");
        Assert(Find<ComboBox>(view, "PolyTargetCombo").Items.Count == 2, "Invalid import preserves existing data");
        var ds = MLDataset.LoadCsv(sample);
        foreach (double lr in new[] { double.NaN, double.PositiveInfinity, -1.0, 1e100 })
        {
            bool rejected = false;
            try { PolynomialRegressionEngine.Train(ds, 1, 2, lr, 1000, 0.8); }
            catch (Exception e) when (e is ArgumentException || e is InvalidOperationException) { rejected = true; }
            Assert(rejected, "Invalid or diverging learning rate is rejected: " + lr);
        }
        var single = Path.Combine(output, "single.csv"); File.WriteAllText(single, "x,y\n1,2\n");
        bool oneRejected = false;
        try { PolynomialRegressionEngine.Train(MLDataset.LoadCsv(single), 1, 2, 0.01, 100, 0.8); }
        catch (ArgumentException) { oneRejected = true; }
        Assert(oneRejected, "One-row dataset is rejected before producing NaN charts");
        window.Width = 1440; window.Height = 900;
        for (int i = 0; i < 7; i++) { window.ShowSection(i); await Settle(window); }
        Assert(true, "All seven workspace sections load");
        await CheckRefinements(window, view);
        await CheckCrossValidation(window, view);
        await CheckPcaAndPageResets(window, view);

    }
    private static async Task CheckCrossValidation(MainWindow window, MLView view)
    {
        var weighted = new CrossValidationReport(true);
        weighted.AddClassification(2, 3, 50, 100);
        weighted.AddClassification(3, 2, 100, 0);
        weighted.Complete();
        Assert(weighted.Rows.Last().TestScore == 60 && weighted.Rows.Last().TrainScore == 80,
            "Combined accuracy weights unequal folds by row count");
        var pooled = new CrossValidationReport();
        pooled.AddRegression(new double[] { 0, 2 }, new double[] { 0, 2 }, new double[] { 0, 2 }, new double[] { 1, 1 });
        pooled.AddRegression(new double[] { 10 }, new double[] { 10 }, new double[] { 10 }, new double[] { 9 });
        pooled.Complete();
        Assert(Math.Abs(pooled.Rows.Last().TestScore - (1 - 3.0 / 56)) < 1e-12,
            "Combined regression R² uses pooled held-out predictions");
        Assert(double.IsNaN(pooled.Rows[1].TestScore), "Singleton fold R² is undefined instead of misleading zero");

        string path = Path.Combine(output, "cv-classification.csv");
        File.WriteAllText(path, "X,Class\n" + string.Join("\n", Enumerable.Range(0, 11).Select(i => $"{i},{(i < 5 ? 0 : 1)}")));
        var ds = MLDataset.LoadCsv(path);
        var cls = new ClassificationEngine { Type = ClassifierType.KNN, Neighbors = 1 }.CrossValidate(ds, 1, new[] { "X" }, 3);
        Assert(cls.Report.Rows.Count == 4 && cls.Report.Rows.Last().TestRows == 11 && cls.Report.Rows.Last().TrainRows == 22,
            "Classification reports every fold and counts every held-out row once");
        Assert(cls.Report.Rows.Take(3).All(r => r.TrainScore == 100), "Training accuracy comes from each fold's own fitted model");
        Assert(cls.Report.Rows.Take(3).Select(r => r.TestScore).SequenceEqual(cls.FoldAccuracy), "Classification table matches engine fold accuracy");
        view.LoadDataset(path); window.ShowSection(3); await Settle(window);
        Find<ComboBox>(view, "CmbClsAlgo").SelectedIndex = 1;
        Find<CheckBox>(view, "ChkClsKFold").IsChecked = true;
        Find<TextBox>(view, "TxtClsFolds").Text = "3";
        await TrainCv("ClsTrain_Click", "ClsCvTable");
        Render(window, "cv-classification");

        var sample = Path.Combine(AppContext.BaseDirectory, "Samples", "curve.csv");
        view.LoadDataset(sample); window.ShowSection(2); await Settle(window);
        Assert(Find<CrossValidationTable>(view, "ClsCvTable").Report == null, "New dataset clears previous cross-validation results");
        Find<ComboBox>(view, "CmbRegAlgo").SelectedIndex = 0;
        Find<CheckBox>(view, "ChkKFold").IsChecked = true;
        await TrainCv("Train_Click", "LinCvTable");
        Find<ComboBox>(view, "CmbRegAlgo").SelectedIndex = 1;
        Find<CheckBox>(view, "PolyUseCV").IsChecked = true;
        await Train(view); await Settle(window);
        var poly = Find<CrossValidationTable>(view, "PolyCvTable").Report;
        Assert(poly != null && poly.Rows.Last().TestRows == 80 && poly.Rows.Last().TestScore > 0.98,
            "Polynomial table shows pooled R² and all 80 held-out predictions");
        window.Width = 1120; window.Height = 700; await Settle(window); Render(window, "cv-polynomial-small");
        Assert(view.BuildExportTables()["cross_validation"].Contains("Combined"), "CSV includes the combined row");
        Find<CheckBox>(view, "PolyUseCV").IsChecked = false;
        await Train(view);
        Assert(Find<CrossValidationTable>(view, "PolyCvTable").Visibility == Visibility.Collapsed
            && !view.BuildExportTables().ContainsKey("cross_validation"), "Holdout training clears and stops exporting the old fold table");
        Find<ComboBox>(view, "CmbRegAlgo").SelectedIndex = 2;
        Find<CheckBox>(view, "ChkAdvKFold").IsChecked = true;
        Find<CheckBox>(view, "ChkAdvCurves").IsChecked = false;
        await TrainCv("AdvTrain_Click", "AdvCvTable");
        Render(window, "cv-advanced");
        typeof(MLView).GetMethod("RefreshDataPreview", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(view, null);
        Assert(new[] { "LinCvTable", "PolyCvTable", "AdvCvTable", "ClsCvTable" }.All(n => Find<CrossValidationTable>(view, n).Report == null),
            "Preprocessing clears all fold tables");

        async Task TrainCv(string handler, string name)
        {
            Click(view, handler);
            var deadline = DateTime.UtcNow.AddSeconds(30);
            while (Find<CrossValidationTable>(view, name).Report == null && DateTime.UtcNow < deadline) await Task.Delay(40);
            await Settle(window);
            var table = Find<CrossValidationTable>(view, name);
            Assert(table.Report != null && table.IsVisible && table.ActualWidth > 400, name + " is populated and visible after training");
            Assert(table.Report!.Rows.Last().Fold == "Combined" && view.BuildExportTables().ContainsKey("cross_validation"), name + " exports fold and combined results");
        }
    }

    private static async Task CheckRefinements(MainWindow window, MLView view)
    {
        for (int tab = 0; tab < 7; tab++)
        {
            window.ShowSection(tab); await Settle(window);
            Assert(Find<Button>(view, "SavePageButton").IsVisible && Find<Button>(view, "SaveChartsButton").IsVisible && Find<Button>(view, "SaveTablesButton").IsVisible,
                "Export actions visible on workspace page " + tab);
        }
        var colorMethod = typeof(MLView).GetMethod("CorrelationTextColor", BindingFlags.Static | BindingFlags.NonPublic)!;
        Assert((Color)colorMethod.Invoke(null, new object[] { Colors.White })! == Colors.Black, "White matrix cells use black numbers");
        Assert((Color)colorMethod.Invoke(null, new object[] { Colors.Blue })! == Colors.White, "Dark blue matrix cells use white numbers");
        Assert((Color)colorMethod.Invoke(null, new object[] { Colors.Red })! == Colors.Black, "Red matrix cells use the higher-contrast black text");

        string path = Path.Combine(output, "unsorted.csv");
        File.WriteAllText(path, "X,Y,Y2\n3,30,300\n1,10,100\n2,20,200\n");
        Assert(view.LoadDataset(path), "Unsorted visualization fixture loads");
        window.ShowSection(6); await Settle(window);
        Find<ComboBox>(view, "VizY2Combo").SelectedIndex = 3;
        Click(view, "VizPlot_Click"); await Settle(window);
        var chart = Find<CartesianChart>(view, "VizCartesian");
        double?[] Xs(int series) => chart.Series.ElementAt(series).Values.Cast<LiveChartsCore.Defaults.ObservablePoint>().Select(p => p.X).ToArray();
        double?[] Ys(int series) => chart.Series.ElementAt(series).Values.Cast<LiveChartsCore.Defaults.ObservablePoint>().Select(p => p.Y).ToArray();
        Assert(Xs(0).SequenceEqual(new double?[] { 3, 1, 2 }), "Line chart preserves source order by default");
        Click(view, "VizSort_Click"); await Settle(window);
        Assert(Xs(0).SequenceEqual(new double?[] { 1, 2, 3 }), "Sort button orders X ascending");
        Assert(Ys(0).SequenceEqual(new double?[] { 10, 20, 30 }) && Ys(1).SequenceEqual(new double?[] { 100, 200, 300 }), "Sorting preserves Y and Y2 row pairings");
        var csv = view.BuildExportTables()["visualization_data"];
        Assert(csv.IndexOf("\"1\",\"10\"") < csv.IndexOf("\"3\",\"30\""), "Chart CSV follows the sorted plot order");
        Render(window, "visualization-sorted");
        ExportChart(view, chart, "line-export");
        Click(view, "VizSort_Click"); await Settle(window);
        Assert(Xs(0).SequenceEqual(new double?[] { 3, 1, 2 }), "Restore button recovers original source order");
        var type = Find<ComboBox>(view, "VizChartTypeCombo");
        type.SelectedIndex = 3;
        Click(view, "VizPlot_Click"); await Settle(window);
        Assert(view.BuildExportTables().ContainsKey("pie_data"), "Pie chart has CSV data export");
        ExportChart(view, Find<LiveChartsCore.SkiaSharpView.WPF.PieChart>(view, "VizPie"), "pie-export");
        type.SelectedIndex = 4;
        Click(view, "VizPlot_Click"); await Settle(window);
        Assert(!Find<Button>(view, "VizSortButton").IsEnabled, "Radar retains feature axes and disables numeric sorting");
        Assert(view.BuildExportTables().ContainsKey("radar_data"), "Radar chart has CSV data export");
        ExportChart(view, Find<LiveChartsCore.SkiaSharpView.WPF.PolarChart>(view, "VizPolar"), "radar-export");
        window.ShowSection(0); await Settle(window);
        Assert(Find<Button>(view, "ImportDatasetButton").IsVisible, "Dataset page contains the import action");
        Assert(view.BuildExportTables().ContainsKey("correlation_matrix"), "Correlation matrix has CSV export");
        var datasetCsv = view.BuildExportTables()["dataset"];
        Assert(datasetCsv.IndexOf("\"3\",\"30\"") < datasetCsv.IndexOf("\"1\",\"10\""), "Visualization sorting leaves the underlying dataset unchanged");
        ExportChart(view, Find<Canvas>(view, "CorrCanvas"), "correlation-export");
        var exported = new BitmapImage(new Uri(Path.Combine(output, "correlation-export.png")));
        var pixel = new byte[4];
        exported.CopyPixels(new Int32Rect(2, 2, 1, 1), pixel, 4, 0);
        Assert(pixel[3] == 255, "Matrix export includes its full canvas above the scroll viewport");
        int blueRibbonPixels = 0;
        int stride = exported.PixelWidth * 4;
        var pixels = new byte[stride * exported.PixelHeight];
        exported.CopyPixels(pixels, stride, 0);
        for (int y = 0; y < exported.PixelHeight; y++)
            for (int x = exported.PixelWidth - 120 * 4; x < exported.PixelWidth; x++)
            {
                int offset = y * stride + x * 4;
                if (pixels[offset] > 200 && pixels[offset + 1] < 100 && pixels[offset + 2] < 100 && pixels[offset + 3] > 200) blueRibbonPixels++;
            }
        Assert(blueRibbonPixels > 100, "Correlation export includes the blue minimum end of the ribbon");
        Render(window, "dataset");
        window.ShowSection(-1); await Settle(window);
        var home = (WelcomeView)((Frame)window.FindName("MainFrame")).Content;
        Assert(((Image)home.FindName("CoverLogo")).Source != null, "Cover loads bundled ibex logo");
        var buttons = Descendants(home).OfType<Button>().Select(button => button.Content?.ToString() ?? "").ToArray();
        Assert(!buttons.Any(label => label.Contains("Import") || label.Contains("demo") || label.Contains("Save cover")), "Home has no import, demo or save-cover buttons");
        Render(window, "home");
    }
    private static System.Collections.Generic.IEnumerable<DependencyObject> Descendants(DependencyObject root)
    {
        for (int i = 0; i < VisualTreeHelper.GetChildrenCount(root); i++)
        {
            var child = VisualTreeHelper.GetChild(root, i);
            yield return child;
            foreach (var descendant in Descendants(child)) yield return descendant;
        }
    }
    private static void CheckReleaseRuntime()
    {
        Assert(typeof(object).Assembly.Location.StartsWith(AppContext.BaseDirectory, StringComparison.OrdinalIgnoreCase), "Self-contained test uses its bundled .NET runtime");
        var data = new Dataset
        {
            X = Enumerable.Range(0, 80).Select(i => new float[] { i / 10f, (i % 7) / 7f }).ToArray(),
            y = Enumerable.Range(0, 80).Select(i => 2f + i / 5f).ToArray()
        };
        foreach (var kind in new[] { RegModelType.FastTree, RegModelType.LightGBM, RegModelType.OGD, RegModelType.XGBoost })
        {
            using var model = RegModelFactory.Create(kind);
            model.SetHyperparameters(new System.Collections.Generic.Dictionary<string, double>
            {
                ["numberOfIterations"] = 10, ["numberOfTrees"] = 10,
                ["numRounds"] = 10, ["minExamplesPerLeaf"] = 2
            });
            model.Train(data);
            var predictions = model.Predict(data.X);
            Assert(predictions.Length == 80 && predictions.All(float.IsFinite), kind + " trains and predicts with packaged dependencies");
            var modelPath = Path.Combine(output, kind + ".aimodel");
            RegModelIO.Save(model, kind, new[] { "X1", "X2" }, modelPath);
            using var restored = RegModelIO.Load(modelPath, out _, out _);
            Assert(restored.Predict(data.X).Zip(predictions).All(pair => Math.Abs(pair.First - pair.Second) < 0.001f), kind + " saves and reloads its model");
        }
        // CoreCLR is built into the Windows single-file host; the core library location is checked above.
        foreach (string name in new[] { "vcomp140.dll", "msvcp140.dll", "vcruntime140.dll" })
        {
            var module = System.Diagnostics.Process.GetCurrentProcess().Modules.Cast<System.Diagnostics.ProcessModule>().First(m => m.ModuleName.Equals(name, StringComparison.OrdinalIgnoreCase));
            Assert(module.FileName.StartsWith(AppContext.BaseDirectory, StringComparison.OrdinalIgnoreCase), name + " loads from the app folder");
        }
    }
    private static void ExportChart(MLView view, FrameworkElement chart, string name)
    {
        string path = Path.Combine(output, name + ".png");
        typeof(MLView).GetMethod("MlExportChart", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(view, new object[] { chart, path });
        Assert(File.Exists(path) && new FileInfo(path).Length > 1000, name + " produces a PNG file");
    }

}

