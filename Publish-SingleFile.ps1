param([string]$OutputDirectory, [string]$VCRedistPath)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
& dotnet restore (Join-Path $projectRoot 'MLStudio.csproj') -r win-x64
if ($LASTEXITCODE -ne 0) { throw 'Package restore failed.' }
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $projectRoot 'release-single' }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$buildRoot = Join-Path $projectRoot ('obj\single-file\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$extras = Join-Path $buildRoot 'extras'
$stage = Join-Path $buildRoot 'publish'
New-Item -ItemType Directory -Force $extras,$stage | Out-Null
foreach ($notice in 'LICENSE','NOTICE','THIRD_PARTY_NOTICES.md') {
    Copy-Item -LiteralPath (Join-Path $projectRoot $notice) -Destination $extras
}
if (-not $VCRedistPath) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install the Visual Studio C++ build tools, or supply -VCRedistPath with the MSVC x64 redistributable folder.' }
    $vsPath = & $vswhere -latest -products '*' -property installationPath | Select-Object -First 1
    if (-not $vsPath -or -not (Test-Path -LiteralPath (Join-Path $vsPath 'VC\Redist\MSVC'))) { throw 'MSVC redistributables were not found. Install the C++ build tools or specify -VCRedistPath.' }
    $version = Get-ChildItem -LiteralPath (Join-Path $vsPath 'VC\Redist\MSVC') -Directory |
        Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' } | Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    $VCRedistPath = Join-Path $version.FullName 'x64'
}
foreach ($pattern in '*.CRT','*.OpenMP') {
    $runtime = Get-ChildItem -LiteralPath $VCRedistPath -Directory | Where-Object { $_.Name -like $pattern } | Select-Object -First 1
    if (-not $runtime) { throw "Missing native redistributable directory: $pattern" }
    Get-ChildItem -LiteralPath $runtime.FullName -Filter '*.dll' -File | Copy-Item -Destination $extras
}
foreach ($name in 'msvcp140.dll','vcruntime140.dll','vcruntime140_1.dll','vcomp140.dll') {
    if (-not (Test-Path -LiteralPath (Join-Path $extras $name))) { throw "Missing native runtime: $name" }
}
# Bundle available dependency notices alongside the native runtime files.
$notices = Join-Path $extras 'ThirdPartyNotices'
New-Item -ItemType Directory -Force $notices | Out-Null
$assets = Get-Content -LiteralPath (Join-Path $projectRoot 'obj\project.assets.json') -Raw | ConvertFrom-Json
foreach ($package in $assets.libraries.PSObject.Properties) {
    if ($package.Value.type -ne 'package') { continue }
    foreach ($packageRoot in $assets.packageFolders.PSObject.Properties.Name) {
        $folder = Join-Path $packageRoot $package.Value.path
        if (-not (Test-Path -LiteralPath $folder)) { continue }
        $files = Get-ChildItem -LiteralPath $folder -File | Where-Object { $_.Name -match '(?i)license|notice|copying|copyright' }
        if ($files) {
            $destination = Join-Path $notices ($package.Name -replace '/', '-')
            New-Item -ItemType Directory -Force $destination | Out-Null
            $files | Copy-Item -Destination $destination
        }
        break
    }
}
& dotnet publish (Join-Path $projectRoot 'MLStudio.csproj') -p:PublishProfile=WindowsSingleFile "-p:SingleFileExtrasDirectory=$extras" -p:SuppressTfmSupportBuildWarnings=true -o $stage -v minimal
if ($LASTEXITCODE -ne 0) { throw 'Single-file publish failed; existing release was not changed.' }
$files = @(Get-ChildItem -LiteralPath $stage -File -Recurse)
if ($files.Count -ne 1 -or $files[0].Name -ne 'IBEX.MLStudio.exe') { throw "Expected exactly one executable in $stage. Existing release was not changed." }
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$target = Join-Path $OutputDirectory 'IBEX.MLStudio.exe'
if (Test-Path -LiteralPath $target) { Copy-Item -LiteralPath $target -Destination (Join-Path $buildRoot 'previous-IBEX.MLStudio.exe') }
Copy-Item -LiteralPath $files[0].FullName -Destination $target -Force
Write-Host "SINGLE_EXE=$target"
Write-Host "BUILD_ROOT=$buildRoot"
Write-Host "SHA256=$((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash)"
