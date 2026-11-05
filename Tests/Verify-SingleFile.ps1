param([Parameter(Mandatory=$true)][string]$Executable, [Parameter(Mandatory=$true)][string]$VerificationDirectory)
$ErrorActionPreference = 'Stop'
$Executable = [IO.Path]::GetFullPath($Executable)
$VerificationDirectory = [IO.Path]::GetFullPath($VerificationDirectory)
New-Item -ItemType Directory -Force $VerificationDirectory | Out-Null
$env:DOTNET_BUNDLE_EXTRACT_BASE_DIR = Join-Path $VerificationDirectory 'bundle'
$env:DOTNET_ROOT = Join-Path $VerificationDirectory 'no-shared-dotnet'
$env:DOTNET_ROOT_X64 = $env:DOTNET_ROOT
$env:DOTNET_MULTILEVEL_LOOKUP = '0'
$env:COREHOST_TRACE = '1'
$env:COREHOST_TRACEFILE = Join-Path $VerificationDirectory 'host-trace.txt'
$process = Start-Process -FilePath $Executable -WorkingDirectory $VerificationDirectory -WindowStyle Hidden -PassThru
try {
    $deadline = (Get-Date).AddSeconds(25)
    do {
        Start-Sleep -Milliseconds 500
        $process.Refresh()
        if ($process.HasExited) { throw "Single-file app exited during startup: $($process.ExitCode)" }
    } while ($process.MainWindowHandle -eq 0 -and (Get-Date) -lt $deadline)
    if ($process.MainWindowHandle -eq 0) { throw 'Single-file app did not create its main window.' }
    if ($process.MainWindowTitle -ne 'IBEX Machine Learning Studio') { throw "Unexpected startup window: $($process.MainWindowTitle)" }
    # The Windows singlefilehost statically embeds CoreCLR; no coreclr.dll module is required.
    $trace = Get-Content -LiteralPath $env:COREHOST_TRACEFILE -Raw
    $baseMatch = [regex]::Matches($trace, '(?m)^Property APP_CONTEXT_BASE_DIRECTORY = (.+)\r?$') | Select-Object -Last 1
    if (-not $baseMatch) { throw 'Missing managed-runtime startup evidence.' }
    $bundle = $baseMatch.Groups[1].Value.Trim()
    if (-not $bundle.StartsWith($env:DOTNET_BUNDLE_EXTRACT_BASE_DIR, [StringComparison]::OrdinalIgnoreCase)) { throw 'App base is outside its extracted bundle.' }
    Write-Host "PASS: Single EXE starts with bundled .NET: $bundle"
    foreach ($name in 'PresentationFramework.dll','libSkiaSharp.dll','lib_lightgbm.dll','xgboost.dll','vcomp140.dll','msvcp140.dll','vcruntime140.dll','IBEX.MLStudio.dll','Samples\curve.csv') {
        if (-not (Test-Path -LiteralPath (Join-Path $bundle $name))) { throw "Missing bundled dependency: $name" }
    }
    $config = Get-Content -LiteralPath (Join-Path $bundle 'IBEX.MLStudio.runtimeconfig.json') -Raw | ConvertFrom-Json
    if ($config.runtimeOptions.framework -or $config.runtimeOptions.frameworks -or $config.runtimeOptions.includedFrameworks.Count -lt 2) { throw 'Runtime config is not self-contained.' }
    $bundle | Set-Content -LiteralPath (Join-Path $VerificationDirectory 'bundle-path.txt')
    Write-Host 'PASS: Required native runtimes, WPF and content files are inside the EXE.'
}
finally {
    if (-not $process.HasExited) {
        $null = $process.CloseMainWindow()
        if (-not $process.WaitForExit(5000)) { $process.Kill(); $process.WaitForExit() }
    }
}
