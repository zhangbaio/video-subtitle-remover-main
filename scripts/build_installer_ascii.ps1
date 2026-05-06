param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("rtx30", "rtx40", "rtx50")]
    [string]$BackendPreset,

    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$VenvPath,
    [string]$InnoSetupPath = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    [switch]$Clean,
    [switch]$SkipQpt,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-BackendConfig([string]$Preset, [string]$Root) {
    switch ($Preset) {
        "rtx30" {
            return @{
                Label           = "RTX 30 / CUDA 11.8"
                TorchSuffix     = "+cu118"
                VenvCandidates  = @(".venv-build-cu118")
                BuildOutputDir  = Join-Path $Root "vsr_out_cu118"
                InstallerScript = Join-Path $Root "installer-cu118-ascii.iss"
                InstallerOutDir = Join-Path $Root "installer-dist-cu118-ascii"
            }
        }
        "rtx40" {
            return @{
                Label           = "RTX 40 / CUDA 12.6"
                TorchSuffix     = "+cu126"
                VenvCandidates  = @(".venv-build-cu126", ".venv-build-cu128")
                BuildOutputDir  = Join-Path $Root "vsr_out_cu126_clean"
                InstallerScript = Join-Path $Root "installer-cu126-ascii-clean.iss"
                InstallerOutDir = Join-Path $Root "installer-dist-cu126-ascii"
            }
        }
        "rtx50" {
            return @{
                Label           = "RTX 50 / CUDA 12.8"
                TorchSuffix     = "+cu128"
                VenvCandidates  = @(".venv-build-cu128")
                BuildOutputDir  = Join-Path $Root "vsr_out_cu128_clean"
                InstallerScript = Join-Path $Root "installer-cu128-ascii.iss"
                InstallerOutDir = Join-Path $Root "installer-dist-cu128-ascii"
            }
        }
    }
}

function Resolve-Venv([hashtable]$Config, [string]$Root, [string]$ExplicitVenv) {
    if ($ExplicitVenv) {
        return [IO.Path]::GetFullPath($ExplicitVenv)
    }

    foreach ($candidate in $Config.VenvCandidates) {
        $full = Join-Path $Root $candidate
        if (Test-Path -LiteralPath $full) {
            return $full
        }
    }

    throw "No matching build venv was found for $($Config.Label). Use -VenvPath or prepare the expected .venv-build directory."
}

function Invoke-Process([string]$Exe, [string[]]$Arguments) {
    Write-Host "$Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE."
    }
}

function Get-TorchVersion([string]$PythonExe) {
    $result = & $PythonExe -c "import torch; print(torch.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to read torch.__version__. Check that the build venv is healthy."
    }
    return ($result | Select-Object -First 1).Trim()
}

function Ensure-Path([string]$Path, [string]$Message) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw $Message
    }
}

function New-PatchedInstallerScript(
    [string]$TemplatePath,
    [string]$ReleaseDir,
    [string]$InstallerOutDir,
    [string]$BackendPreset
) {
    $generatedDir = Join-Path $PSScriptRoot "_generated"
    if (-not (Test-Path -LiteralPath $generatedDir)) {
        New-Item -ItemType Directory -Path $generatedDir | Out-Null
    }

    $generatedPath = Join-Path $generatedDir ("installer-{0}.iss" -f $BackendPreset)
    $content = Get-Content -LiteralPath $TemplatePath -Raw
    $content = $content -replace '(?m)^#define SourceDir ".*"$', ('#define SourceDir "' + $ReleaseDir + '"')
    $content = $content -replace '(?m)^OutputDir=.*$', ('OutputDir=' + $InstallerOutDir)
    Set-Content -LiteralPath $generatedPath -Value $content -Encoding UTF8
    return $generatedPath
}

$RepoRoot = [IO.Path]::GetFullPath($RepoRoot)
$Config = Get-BackendConfig -Preset $BackendPreset -Root $RepoRoot
$ResolvedVenv = Resolve-Venv -Config $Config -Root $RepoRoot -ExplicitVenv $VenvPath
$PythonExe = Join-Path $ResolvedVenv "Scripts\python.exe"
$QptExe = Join-Path $ResolvedVenv "Scripts\qpt.exe"
$RequirementsPath = Join-Path $RepoRoot "requirements.txt"
$MainFile = Join-Path $RepoRoot "gui.py"
$IconPath = Join-Path $RepoRoot "design\vsr.ico"
$BuildOutputDir = [IO.Path]::GetFullPath($Config.BuildOutputDir)
$ReleaseDir = Join-Path $BuildOutputDir "Release"
$InstallerOutDir = [IO.Path]::GetFullPath($Config.InstallerOutDir)

Write-Step "Validate environment"
Ensure-Path -Path $RepoRoot -Message "Repo root not found: $RepoRoot"
Ensure-Path -Path $PythonExe -Message "Build venv python not found: $PythonExe"
Ensure-Path -Path $QptExe -Message "QPT executable not found: $QptExe"
Ensure-Path -Path $RequirementsPath -Message "requirements.txt not found: $RequirementsPath"
Ensure-Path -Path $MainFile -Message "gui.py not found: $MainFile"
Ensure-Path -Path $IconPath -Message "Icon file not found: $IconPath"
Ensure-Path -Path $Config.InstallerScript -Message "Installer template not found: $($Config.InstallerScript)"
Ensure-Path -Path $InnoSetupPath -Message "ISCC.exe not found: $InnoSetupPath"

$TorchVersion = Get-TorchVersion -PythonExe $PythonExe
if ($TorchVersion -notlike "*$($Config.TorchSuffix)") {
    throw "Torch version mismatch. Current: $TorchVersion. Expected suffix: $($Config.TorchSuffix) for $($Config.Label)."
}

Write-Host "Backend:      $($Config.Label)"
Write-Host "RepoRoot:     $RepoRoot"
Write-Host "VenvPath:     $ResolvedVenv"
Write-Host "TorchVersion: $TorchVersion"
Write-Host "BuildOutput:  $BuildOutputDir"
Write-Host "InstallerIss: $($Config.InstallerScript)"
Write-Host "InstallerOut: $InstallerOutDir"

if ($Clean) {
    Write-Step "Clean old outputs"
    if (Test-Path -LiteralPath $BuildOutputDir) {
        Remove-Item -LiteralPath $BuildOutputDir -Recurse -Force
    }
    if (Test-Path -LiteralPath $InstallerOutDir) {
        Remove-Item -LiteralPath $InstallerOutDir -Recurse -Force
    }
}

if (-not $SkipQpt) {
    Write-Step "Run QPT"
    Invoke-Process -Exe $QptExe -Arguments @(
        "-f", $RepoRoot,
        "-p", "gui.py",
        "-s", $BuildOutputDir,
        "-r", $RequirementsPath,
        "-h", "false",
        "-i", $IconPath
    )
}

Write-Step "Validate release output"
Ensure-Path -Path $ReleaseDir -Message "Release directory not found: $ReleaseDir"
if (-not (Test-Path -LiteralPath (Join-Path $ReleaseDir "VSR.exe"))) {
    Write-Warning "VSR.exe was not found in $ReleaseDir. Please verify the QPT result before shipping."
}
Ensure-Path -Path (Join-Path $ReleaseDir "resources\design\vsr.ico") -Message "Release icon missing: $ReleaseDir\resources\design\vsr.ico"

if (-not $SkipInstaller) {
    Write-Step "Patch Inno Setup script"
    $PatchedIss = New-PatchedInstallerScript `
        -TemplatePath $Config.InstallerScript `
        -ReleaseDir $ReleaseDir `
        -InstallerOutDir $InstallerOutDir `
        -BackendPreset $BackendPreset

    Write-Step "Compile installer"
    Invoke-Process -Exe $InnoSetupPath -Arguments @($PatchedIss)
}

Write-Step "Done"
Write-Host "Build finished. Check: $InstallerOutDir" -ForegroundColor Green
