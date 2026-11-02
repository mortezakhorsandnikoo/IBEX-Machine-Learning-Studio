param(
    [ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version = '1.0.0',
    [string]$VCRedistPath
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$releaseRoot = Join-Path $projectRoot 'releases'
$packageName = "IBEX-MLStudio-$Version-win-x64"
$zipPath = Join-Path $releaseRoot "$packageName.zip"
if (Test-Path -LiteralPath $zipPath) { throw "Release already exists: $zipPath. Choose a new version to preserve it." }
# Always publish into a new staging directory; never delete or overwrite a previous release.
$stageRoot = Join-Path $releaseRoot ("staging\" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$publishDir = Join-Path $stageRoot $packageName
New-Item -ItemType Directory -Force -Path $publishDir | Out-Null
& dotnet publish (Join-Path $projectRoot 'MLStudio.csproj') -p:PublishProfile=WindowsSelfContained "-p:Version=$Version" -o $publishDir -v minimal
if ($LASTEXITCODE -ne 0) { throw 'Self-contained publish failed.' }

# Use Visual Studio's designated redistributable directories, never arbitrary System32 DLLs.
if (-not $VCRedistPath) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Provide -VCRedistPath pointing to the Visual Studio Redist/MSVC/<version>/x64 folder.' }
    $vsPath = (& $vswhere -latest -products '*' -property installationPath | Select-Object -First 1)
    $redistRoot = Join-Path $vsPath 'VC\Redist\MSVC'
    $redistVersion = Get-ChildItem -LiteralPath $redistRoot -Directory | Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' } | Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    $VCRedistPath = Join-Path $redistVersion.FullName 'x64'
}
foreach ($pattern in '*.CRT','*.OpenMP') {
    $runtimeDir = Get-ChildItem -LiteralPath $VCRedistPath -Directory | Where-Object { $_.Name -like $pattern } | Select-Object -First 1
    if (-not $runtimeDir) { throw "Missing Visual C++ redistributable folder: $pattern" }
    Get-ChildItem -LiteralPath $runtimeDir.FullName -File | Copy-Item -Destination $publishDir
}
$config = Get-Content -LiteralPath (Join-Path $publishDir 'IBEX.MLStudio.runtimeconfig.json') -Raw | ConvertFrom-Json
if ($config.runtimeOptions.framework -or $config.runtimeOptions.frameworks -or $config.runtimeOptions.includedFrameworks.Count -lt 2) { throw 'The publish requires a shared .NET installation; refusing to package.' }
foreach ($required in 'IBEX.MLStudio.exe','coreclr.dll','hostfxr.dll','hostpolicy.dll','PresentationFramework.dll','libSkiaSharp.dll','lib_lightgbm.dll','xgboost.dll','msvcp140.dll','vcruntime140.dll','vcruntime140_1.dll','vcomp140.dll') {
    if (-not (Test-Path -LiteralPath (Join-Path $publishDir $required))) { throw "Missing required runtime file: $required" }
}

$noticesDir = Join-Path $publishDir 'ThirdPartyNotices'
New-Item -ItemType Directory -Force -Path $noticesDir | Out-Null
$assets = Get-Content -LiteralPath (Join-Path $projectRoot 'obj\project.assets.json') -Raw | ConvertFrom-Json
$packageRoots = $assets.packageFolders.PSObject.Properties.Name
$inventory = [System.Collections.Generic.List[string]]::new()
$inventory.Add('IBEX Machine Learning Studio - bundled dependency inventory')
$inventory.Add('License identifiers and notices below describe third-party components, not the IBEX application itself.')
foreach ($package in $assets.libraries.PSObject.Properties) {
    if ($package.Value.type -ne 'package') { continue }
    $folder = $null
    foreach ($packageRoot in $packageRoots) {
        $candidate = Join-Path $packageRoot $package.Value.path
        if (Test-Path -LiteralPath $candidate) { $folder = $candidate; break }
    }
    if (-not $folder) { continue }
    $spec = Get-ChildItem -LiteralPath $folder -Filter '*.nuspec' | Select-Object -First 1
    if ($spec) {
        [xml]$xml = Get-Content -LiteralPath $spec.FullName -Raw
        $metadata = $xml.SelectSingleNode("//*[local-name()='metadata']")
        $inventory.Add("`n$($package.Name)`nAuthors: $($metadata.authors)`nLicense: $($metadata.license.InnerText)`nProject: $($metadata.projectUrl)`nLicense URL: $($metadata.licenseUrl)")
    }
    $notices = Get-ChildItem -LiteralPath $folder -File | Where-Object { $_.Name -match '(?i)license|notice|copying|copyright' }
    if ($notices) {
        $target = Join-Path $noticesDir ($package.Name -replace '/','-')
        New-Item -ItemType Directory -Force -Path $target | Out-Null
        $notices | Copy-Item -Destination $target
    }
}
# Runtime packs have their own license and third-party notices.
foreach ($framework in $config.runtimeOptions.includedFrameworks) {
    $id = ($framework.name + '.Runtime.win-x64').ToLowerInvariant()
    foreach ($packageRoot in $packageRoots) {
        $folder = Join-Path $packageRoot "$id\$($framework.version)"
        if (Test-Path -LiteralPath $folder) {
            $target = Join-Path $noticesDir "$id-$($framework.version)"
            New-Item -ItemType Directory -Force -Path $target | Out-Null
            Get-ChildItem -LiteralPath $folder -File | Where-Object { $_.Name -match '(?i)license|notice' } | Copy-Item -Destination $target
            break
        }
    }
}
$inventory.Add("`nMicrosoft Visual C++ runtime (CRT and OpenMP), from the installed Visual Studio redistributable directories.`nhttps://learn.microsoft.com/cpp/windows/redistributing-visual-cpp-files")
$inventory | Set-Content -LiteralPath (Join-Path $noticesDir 'DEPENDENCIES.txt') -Encoding utf8
@"
IBEX Machine Learning Studio $Version - Windows x64 portable edition
Designed by Morteza Khorsand Nikoo
morteza.khorsandnikoo@uibk.ac.at

1. Extract the entire ZIP to a folder on your computer.
2. Open that folder and double-click IBEX.MLStudio.exe.
3. Select Dataset in the sidebar, then Import dataset.

No .NET download, .NET installation, or Visual Studio installation is needed.
The .NET Desktop Runtime and required Visual C++ runtime files are included.
Keep the whole extracted folder together. Do not distribute or move only the EXE.
This package targets 64-bit Windows 10/11 on Intel/AMD PCs, not macOS or Linux.

Support development: https://ko-fi.com/mortezakhorsand
Third-party notices are in ThirdPartyNotices.
"@ | Set-Content -LiteralPath (Join-Path $publishDir 'START-HERE.txt') -Encoding utf8
Compress-Archive -LiteralPath $publishDir -DestinationPath $zipPath -CompressionLevel Optimal
$hash = Get-FileHash -LiteralPath $zipPath -Algorithm SHA256
"$($hash.Hash.ToLowerInvariant())  $packageName.zip" | Set-Content -LiteralPath "$zipPath.sha256" -Encoding ascii
Write-Host "RELEASE_ZIP=$zipPath"
Write-Host "RELEASE_FOLDER=$publishDir"
Write-Host "SHA256=$($hash.Hash)"
