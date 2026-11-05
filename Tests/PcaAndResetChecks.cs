using System.Globalization;
using System.IO;
using System.Reflection;
using System.Windows;
using System.Windows.Controls;
using ArchiAI;
using ArchiAI.ML;
using ArchiAI.Views;
using LiveChartsCore.SkiaSharpView.WPF;

internal static partial class Smoke
{
    private static T Field<T>(MLView view, string name) => (T)typeof(MLView).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(view)!;
    private static async Task Until(Func<bool> predicate, string message)
    {
        var deadline = DateTime.UtcNow.AddSeconds(30);
        while (!predicate() && DateTime.UtcNow < deadline) await Task.Delay(40);
        Assert(predicate(), message);
    }

    private static async Task CheckPcaAndPageResets(MainWindow window, MLView view)
    {
        var inv = CultureInfo.InvariantCulture;
        var rawRows = Enumerable.Range(0, 40).Select(i => new[] { 20.0 + i, 2.0 + (i % 7), 100.0 + 2*i + 3*(i%7) }).ToArray();
        string path = Path.Combine(output, "original-features.csv");
        File.WriteAllText(path, "Area,Height,Response\n" + string.Join("\n", rawRows.Select(row => string.Join(",", row.Select(v => v.ToString("R", inv))))));
        var original = MLDataset.LoadCsv(path);
        foreach (var method in new[] { NormMethod.ZScore, NormMethod.MinMax, NormMethod.MaxAbs })
        {
            var normalization = NormalizationRecipe.ForDataset(original, method);
            var normalized = PreprocessingEngine.NormalizeData(original, method);
            Assert(rawRows.SelectMany((row, i) => row.Select((v, c) => Math.Abs(normalization.Transform(original.Headers[c], v) - normalized.NormalizedData[i][c]))).Max() < 1e-10,
                method + " recipe reproduces normalization exactly");
        }
        var pca = PreprocessingEngine.RunPCA(original, 1, 1, true);
        var recipe = PredictionRecipe.Create(new[] { "PC1" }, "Response", new[] { PreprocessingStep.FromPca(pca) });
        Assert(recipe.InputNames.SequenceEqual(new[] { "Area", "Height" }), "PCA recipe asks for original feature names and excludes target");
        Assert(rawRows.Select((row, i) => Math.Abs(recipe.Transform(row.Take(2).ToArray())[0] - pca.ProjectedData[i][0])).Max() < 1e-10,
            "Prediction projection matches PCA training projection on every row");
        bool rejected = false;
        try { PredictionRecipe.Create(new[] { "Response" }, "PC1", new[] { PreprocessingStep.FromPca(pca) }); }
        catch (InvalidOperationException) { rejected = true; }
        Assert(rejected, "Supervised prediction rejects a PCA component as the target");
        string secondPath = Path.Combine(output, "second-pca.csv");
        File.WriteAllText(secondPath, "PC1,Response\n" + string.Join("\n", pca.ProjectedData.Select((row, i) => row[0].ToString("R", inv) + "," + rawRows[i][2].ToString("R", inv))));
        var secondPca = PreprocessingEngine.RunPCA(MLDataset.LoadCsv(secondPath), 1, 1, true);
        var repeatedRecipe = PredictionRecipe.Create(new[] { "PC1" }, "Response", new[] { PreprocessingStep.FromPca(pca), PreprocessingStep.FromPca(secondPca) });
        Assert(rawRows.Select((row, i) => Math.Abs(repeatedRecipe.Transform(row.Take(2).ToArray())[0] - secondPca.ProjectedData[i][0])).Max() < 1e-10,
            "Repeated PCA retains the full original-feature transformation chain");
        var legacy = new SavedModel { Type = "Linear", TargetName = "Response", FeatureNames = new[] { "Area", "Height" }, Weights = new[] { 2.0, 3.0 }, Bias = 7 };
        ModelIO.Save(legacy, Path.Combine(output, "legacy-no-pca.aimodel"));
        var restoredLegacy = ModelIO.Load(Path.Combine(output, "legacy-no-pca.aimodel"));
        Assert(restoredLegacy.Pipeline == null && restoredLegacy.PredictRaw(new[] { 4.0, 5.0 }) == 30,
            "Existing models without preprocessing metadata keep their original prediction behavior");

        view.LoadDataset(path); window.ShowSection(1); await Settle(window);
        Find<ComboBox>(view, "CmbNormMethod").SelectedIndex = 0;
        Click(view, "Normalize_Click");
        Find<ComboBox>(view, "CmbPcaMode").SelectedIndex = 1;
        Find<TextBox>(view, "TxtPcaValue").Text = "1";
        Find<CheckBox>(view, "ChkPcaHasTarget").IsChecked = true;
        Click(view, "Pca_Click");
        await Until(() => Field<MLDataset>(view, "_dataset").Headers[0] == "PC1", "UI applies PCA with target preserved");
        Click(view, "Normalize_Click"); // also test normalization after PCA
        var prepared = Field<MLDataset>(view, "_dataset");
        var steps = Field<List<PreprocessingStep>>(view, "_preprocessingSteps");
        var pipeline = PredictionRecipe.Create(new[] { "PC1" }, "Response", steps);
        double[] raw = rawRows[8].Take(2).ToArray();
        Assert(Math.Abs(pipeline.Transform(raw)[0] - prepared.Rows[8][0]) < 1e-8,
            "Raw features reproduce normalize → PCA → normalize training inputs");
        Assert(Math.Abs(pipeline.RestoreTarget(prepared.Rows[8][1]) - rawRows[8][2]) < 1e-7,
            "Predicted target converts back to its original units");

        window.ShowSection(2); Find<ComboBox>(view, "CmbRegAlgo").SelectedIndex = 0;
        Find<CheckBox>(view, "ChkKFold").IsChecked = false;
        Click(view, "Train_Click");
        await Until(() => Field<SavedModel>(view, "_linModel")?.Pipeline != null, "Linear model captures its preprocessing recipe");
        var linear = Field<SavedModel>(view, "_linModel");
        Assert(linear.InputNames.SequenceEqual(new[] { "Area", "Height" }) && Field<List<TextBox>>(view, "_linInputs").Count == 2,
            "Linear prediction form uses two original features despite one PCA component");
        TestSavedModel(linear, "linear-pca.aimodel", raw);
        Fill(Field<List<TextBox>>(view, "_linInputs"), raw); Click(view, "LinPredict_Click");
        Assert(Find<TextBlock>(view, "LinPredictResult").Text.StartsWith("Response ="), "Linear prediction labels the original target");

        Find<ComboBox>(view, "CmbRegAlgo").SelectedIndex = 1;
        Find<CheckBox>(view, "PolyUseCV").IsChecked = false;
        await Train(view);
        var polynomial = Field<SavedModel>(view, "_polyModel");
        Assert(polynomial.InputNames.SequenceEqual(new[] { "Area", "Height" }), "Polynomial prediction uses original feature names");
        TestSavedModel(polynomial, "polynomial-pca.aimodel", raw);
        Fill(Field<List<TextBox>>(view, "_polyInputs"), raw); Click(view, "PolyPredict_Click");
        window.Width = 1440; window.Height = 900; await Settle(window);
        Find<StackPanel>(view, "PolyPredictInputs").BringIntoView(); await Settle(window);
        Render(window, "pca-original-feature-prediction");

        Find<ComboBox>(view, "CmbRegAlgo").SelectedIndex = 2;
        Find<CheckBox>(view, "ChkAdvKFold").IsChecked = false;
        Find<CheckBox>(view, "ChkAdvCurves").IsChecked = false;
        Click(view, "AdvTrain_Click");
        await Until(() => Field<IRegressionModel>(view, "_advModel") != null, "Advanced regression trains on PCA data");
        var advanced = Field<IRegressionModel>(view, "_advModel");
        var advPipeline = Field<PredictionRecipe>(view, "_advPipeline");
        Assert(advPipeline.InputNames.SequenceEqual(new[] { "Area", "Height" }) && Field<List<TextBox>>(view, "_advPredictBoxes").Count == 2,
            "Advanced regression form uses original features");
        var advPath = Path.Combine(output, "advanced-pca.aimodel");
        RegModelIO.Save(advanced, RegModelType.NormalEquation, advPipeline.ModelFeatures, advPath, advPipeline);
        using (var loaded = RegModelIO.Load(advPath, out _, out _, out var restored))
        {
            var before = advPipeline.RestoreTarget(advanced.Predict(new[] { advPipeline.Transform(raw).Select(v => (float)v).ToArray() })[0]);
            var after = restored.RestoreTarget(loaded.Predict(new[] { restored.Transform(raw).Select(v => (float)v).ToArray() })[0]);
            Assert(Math.Abs(before - after) < 1e-6, "Advanced model save/load preserves PCA and target units");
        }
        Fill(Field<List<TextBox>>(view, "_advPredictBoxes"), raw); Click(view, "AdvPredict_Click");
        Assert(Find<TextBlock>(view, "AdvPredictResult").Text.StartsWith("Response:"), "Advanced prediction labels original target");

        window.ShowSection(3); await Settle(window);
        Find<ComboBox>(view, "CmbClsAlgo").SelectedIndex = 1;
        Find<CheckBox>(view, "ChkClsKFold").IsChecked = false;
        Click(view, "ClsTrain_Click");
        await Until(() => Field<ClassificationResult>(view, "_clsResult")?.Pipeline != null, "Classification captures its preprocessing pipeline");
        var classifier = Field<ClassificationResult>(view, "_clsResult");
        Assert(classifier.InputNames.SequenceEqual(new[] { "Area", "Height" }), "Classification prediction uses original features");
        string clsPath = Path.Combine(output, "classifier-pca.aiclf");
        ClassificationIO.Save(classifier, classifier.FeatureNames, classifier.TargetName, clsPath);
        var loadedCls = ClassificationIO.Load(clsPath, out _, out _);
        Assert(Math.Abs(classifier.PredictRaw(raw) - loadedCls.PredictRaw(raw)) < 1e-9, "Classifier save/load preserves original-input prediction");

        window.ShowSection(4); Click(view, "ClustRun_Click");
        await Until(() => Field<ClusterResult>(view, "_clustResult") != null, "Clustering trains on prepared data");
        Assert(Field<PredictionRecipe>(view, "_clustPipeline").InputNames.SequenceEqual(original.Headers),
            "Clustering assignment uses original columns");
        Fill(Field<List<TextBox>>(view, "_clustPredictBoxes"), rawRows[8]); Click(view, "ClustPredict_Click");
        Assert(Find<TextBlock>(view, "ClustPredictResult").Text.StartsWith("Assigned to cluster:"), "Clustering applies PCA internally for assignment");

        // Reset each page and verify the trained models on other pages remain usable.
        bool regressionReset = false;
        foreach (int page in new[] { 1, 0, 6, 2, 5, 4, 3 })
        {
            window.ShowSection(page); await Settle(window);
            Assert(Find<Button>(view, "ResetPageButton").IsVisible, "Reset action visible on workspace page " + page);
            var datasetBefore = Field<MLDataset>(view, "_dataset");
            var stepsBefore = Field<List<PreprocessingStep>>(view, "_preprocessingSteps").Count;
            Click(view, "ResetPage_Click"); await Settle(window);
            Assert(ReferenceEquals(datasetBefore, Field<MLDataset>(view, "_dataset")) && stepsBefore == Field<List<PreprocessingStep>>(view, "_preprocessingSteps").Count,
                "Page " + page + " reset preserves shared data and preprocessing lineage");
            Assert(view.BuildExportTables().Count == 0, "Page " + page + " reset removes its exported results");
            if (page == 2) regressionReset = true;
            if (!regressionReset)
                Assert(ReferenceEquals(linear, Field<SavedModel>(view, "_linModel")) && double.IsFinite(linear.PredictRaw(raw)),
                    "Page " + page + " reset preserves regression model and original-input predictions");
            if (page != 3)
                Assert(ReferenceEquals(classifier, Field<ClassificationResult>(view, "_clsResult")),
                    "Page " + page + " reset preserves the classification model");
        }
        Assert(Field<SavedModel>(view, "_linModel") == null && Field<SavedModel>(view, "_polyModel") == null
            && Field<IRegressionModel>(view, "_advModel") == null, "Regression reset clears all algorithm models on that page");
        window.ShowSection(0); await Settle(window);
        Assert(Find<Canvas>(view, "CorrCanvas").Children.Count == 0 && Find<DataGrid>(view, "DataPreview").ItemsSource == null,
            "Reset dataset page stays empty after navigation and layout changes");
        Render(window, "dataset-after-page-reset");
        window.ShowSection(3); Click(view, "ClsTrain_Click"); Click(view, "ResetPage_Click");
        await Task.Delay(1200);
        Assert(Field<ClassificationResult>(view, "_clsResult") == null && Find<TextBox>(view, "TxtClsResults").Text == "",
            "Reset discards a classification result that finishes in the background");

        void TestSavedModel(SavedModel model, string file, double[] input)
        {
            string savedPath = Path.Combine(output, file); ModelIO.Save(model, savedPath);
            var loaded = ModelIO.Load(savedPath);
            Assert(loaded.InputNames.SequenceEqual(model.InputNames) && Math.Abs(loaded.PredictRaw(input) - model.PredictRaw(input)) < 1e-9,
                model.Type + " save/load preserves original features, PCA, and target units");
        }
        void Fill(List<TextBox> boxes, double[] values)
        { for (int i = 0; i < values.Length; i++) boxes[i].Text = values[i].ToString("R", inv); }
    }
}
