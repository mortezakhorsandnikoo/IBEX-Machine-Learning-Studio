# Third-party dependencies

IBEX uses third-party libraries restored through NuGet. Their licenses are separate from the license for IBEX's own source. The definitive versions are declared in `MLStudio.csproj` and resolved in `obj/project.assets.json` after restore.

Direct dependencies include:

- CommunityToolkit.Mvvm
- LiveChartsCore.SkiaSharpView.WPF (and SkiaSharp-related dependencies)
- MathNet.Numerics
- Microsoft.ML, Microsoft.ML.FastTree and Microsoft.ML.LightGbm
- Microsoft.ML.OnnxRuntime
- XGBoostSharp-cpu

To inspect all direct and transitive versions, run `dotnet list MLStudio.csproj package --include-transitive`. Review the packages' license metadata and notices when redistributing binaries. `Publish-SingleFile.ps1` includes available top-level license and notice files from restored packages inside the bundle; this does not replace reviewing applicable distribution terms.

The publishing script also uses Microsoft Visual C++ redistributable files from an installed build toolchain. It does not include those binaries in this source repository.

Optional PDF authoring uses ReportLab and locally installed Windows Calibri fonts. Font files are not included in the source repository.
