param(
    [string]$InstallDir = "",
    [ValidateSet("auto", "cpu", "directml", "cu118", "cu126", "cu128")]
    [string]$Backend = "auto",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$PythonVersion = "3.10.11"
$PythonInstallerName = "python-$PythonVersion-amd64.exe"
$PythonInstallerUrl = "https://www.python.org/ftp/python/$PythonVersion/$PythonInstallerName"
$InstallMarkerName = ".vsr-online-install.json"
$InstallProductId = "video-subtitle-remover-online"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Resolve-ScriptRoot {
    if ($PSScriptRoot) {
        return (Resolve-Path -LiteralPath $PSScriptRoot).Path
    }
    return (Get-Location).Path
}

function Resolve-ProjectSourceDir {
    $scriptRoot = Resolve-ScriptRoot
    $candidate = Split-Path -Parent $scriptRoot
    if (Test-Path -LiteralPath (Join-Path $candidate "gui.py")) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }
    if (Test-Path -LiteralPath (Join-Path $scriptRoot "gui.py")) {
        return $scriptRoot
    }
    throw "Cannot find VSR source directory. Put this script under the project's scripts folder or project root."
}

function Get-InstallMarkerPath {
    param([string]$InstallRoot)
    return (Join-Path $InstallRoot $InstallMarkerName)
}

function Assert-SafeInstallRootPath {
    param([string]$InstallRoot)
    if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
        throw "Install directory cannot be empty."
    }

    $fullPath = [System.IO.Path]::GetFullPath($InstallRoot).TrimEnd("\")
    $rootPath = [System.IO.Path]::GetPathRoot($fullPath).TrimEnd("\")
    if ($fullPath -eq $rootPath) {
        throw "Refusing to use drive root as install directory: $InstallRoot"
    }

    $blockedPaths = @(
        $env:USERPROFILE,
        $env:LOCALAPPDATA,
        $env:APPDATA,
        $env:PROGRAMFILES,
        ${env:ProgramFiles(x86)}
    )
    foreach ($blockedPath in $blockedPaths) {
        if ([string]::IsNullOrWhiteSpace($blockedPath)) {
            continue
        }
        $blockedFullPath = [System.IO.Path]::GetFullPath($blockedPath).TrimEnd("\")
        if ($fullPath -ieq $blockedFullPath) {
            throw "Refusing to use broad system or profile directory as install directory: $InstallRoot"
        }
    }
}

function Test-DirectoryEmpty {
    param([string]$Path)
    if (!(Test-Path -LiteralPath $Path)) {
        return $true
    }
    return $null -eq (Get-ChildItem -LiteralPath $Path -Force | Select-Object -First 1)
}

function Test-OwnedInstallRoot {
    param([string]$InstallRoot)
    $markerPath = Get-InstallMarkerPath -InstallRoot $InstallRoot
    if (!(Test-Path -LiteralPath $markerPath)) {
        return $false
    }
    try {
        $marker = Get-Content -Raw -LiteralPath $markerPath | ConvertFrom-Json
        return $marker.productId -eq $InstallProductId
    } catch {
        return $false
    }
}

function Assert-OwnedInstallRoot {
    param([string]$InstallRoot)
    if (!(Test-OwnedInstallRoot -InstallRoot $InstallRoot)) {
        throw "Refusing to delete install directory because it is not marked as a VSR Online installation: $InstallRoot"
    }
}

function Write-InstallMarker {
    param([string]$InstallRoot)
    $markerPath = Get-InstallMarkerPath -InstallRoot $InstallRoot
    $marker = [ordered]@{
        productId = $InstallProductId
        createdBy = "install_vsr_online.ps1"
        createdAt = (Get-Date).ToString("o")
        projectDirName = "video-subtitle-remover-main"
        pythonVersion = $PythonVersion
    }
    Set-Content -LiteralPath $markerPath -Value ($marker | ConvertTo-Json) -Encoding ASCII
}

function Copy-ProjectSource {
    param(
        [string]$SourceDir,
        [string]$TargetDir
    )
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    $excludeDirs = @(
        ".git",
        ".venv",
        ".venv-build-cu126",
        ".venv-build-cu118",
        "installer-dist",
        "installer-dist-cu126-ascii",
        "__pycache__"
    )
    $excludeFiles = @("*.pyc", "*.pyo")
    $robocopyArgs = @(
        $SourceDir,
        $TargetDir,
        "/E",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP",
        "/XD"
    ) + $excludeDirs + @("/XF") + $excludeFiles
    & robocopy @robocopyArgs | Out-Null
    if ($LASTEXITCODE -gt 7) {
        throw "Failed to copy project files. robocopy exit code: $LASTEXITCODE"
    }
}

function Get-NvidiaGpuName {
    try {
        $names = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $names) {
            return (($names | Select-Object -First 1) -as [string]).Trim()
        }
    } catch {
    }
    return ""
}

function Resolve-Backend {
    param([string]$RequestedBackend)
    if ($RequestedBackend -ne "auto") {
        return $RequestedBackend
    }
    $gpuName = Get-NvidiaGpuName
    if ($gpuName -match "RTX\s*50") {
        return "cu128"
    }
    if ($gpuName -match "RTX\s*40") {
        return "cu126"
    }
    if ($gpuName -match "RTX\s*30|RTX\s*20") {
        return "cu118"
    }
    return "cpu"
}

function Install-EmbeddedPython {
    param(
        [string]$PythonDir,
        [string]$DownloadDir
    )
    $pythonExe = Join-Path $PythonDir "python.exe"
    if (Test-Path -LiteralPath $pythonExe) {
        return $pythonExe
    }

    New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
    $installerPath = Join-Path $DownloadDir $PythonInstallerName
    if (!(Test-Path -LiteralPath $installerPath)) {
        Write-Step "Downloading Python $PythonVersion"
        Invoke-WebRequest -Uri $PythonInstallerUrl -OutFile $installerPath
    }

    Write-Step "Installing local Python $PythonVersion"
    $args = @(
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=0",
        "Include_pip=1",
        "Include_test=0",
        "TargetDir=$PythonDir"
    )
    $process = Start-Process -FilePath $installerPath -ArgumentList $args -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Python installer failed with exit code $($process.ExitCode)."
    }
    if (!(Test-Path -LiteralPath $pythonExe)) {
        throw "Python install finished, but python.exe was not found at $pythonExe"
    }
    return $pythonExe
}

function Install-PythonPackages {
    param(
        [string]$VenvPython,
        [string]$ProjectDir,
        [string]$ResolvedBackend
    )
    Write-Step "Upgrading pip"
    & $VenvPython -m pip install --upgrade pip setuptools wheel

    Write-Step "Installing PaddlePaddle"
    & $VenvPython -m pip install paddlepaddle==3.0.0

    Write-Step "Installing VSR requirements"
    & $VenvPython -m pip install -r (Join-Path $ProjectDir "requirements.txt")

    Write-Step "Installing inference backend: $ResolvedBackend"
    switch ($ResolvedBackend) {
        "cu118" {
            & $VenvPython -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
        }
        "cu126" {
            & $VenvPython -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
        }
        "cu128" {
            & $VenvPython -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
        }
        "directml" {
            & $VenvPython -m pip install torch==2.4.1 torchvision==0.19.1
            & $VenvPython -m pip install torch_directml==0.2.5.dev240914
        }
        default {
            & $VenvPython -m pip install torch==2.7.0 torchvision==0.22.0
        }
    }
}

function Write-RunScript {
    param([string]$ProjectDir)
    $runPath = Join-Path $ProjectDir "run_vsr.bat"
    $content = @'
@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" gui.py
pause
'@
    Set-Content -LiteralPath $runPath -Value $content -Encoding ASCII
}

function New-DesktopShortcut {
    param([string]$ProjectDir)
    $desktop = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = Join-Path $desktop "VSR.lnk"
    $wshShell = New-Object -ComObject WScript.Shell
    $shortcut = $wshShell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = Join-Path $ProjectDir "run_vsr.bat"
    $shortcut.WorkingDirectory = $ProjectDir
    $iconPath = Join-Path $ProjectDir "design\vsr.ico"
    if (Test-Path -LiteralPath $iconPath) {
        $shortcut.IconLocation = $iconPath
    }
    $shortcut.Save()
}

if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = Join-Path $env:LOCALAPPDATA "Programs\VSR_Online"
}

$sourceDir = Resolve-ProjectSourceDir
$installRoot = [System.IO.Path]::GetFullPath($InstallDir)
$projectDir = Join-Path $installRoot "video-subtitle-remover-main"
$pythonDir = Join-Path $installRoot "Python310"
$downloadDir = Join-Path $installRoot "downloads"
$venvDir = Join-Path $projectDir ".venv"

Write-Step "Preparing install directory: $installRoot"
Assert-SafeInstallRootPath -InstallRoot $installRoot
if ($Force -and (Test-Path -LiteralPath $installRoot)) {
    if (Test-DirectoryEmpty -Path $installRoot) {
        Remove-Item -LiteralPath $installRoot -Force
    } else {
        Assert-OwnedInstallRoot -InstallRoot $installRoot
        Remove-Item -LiteralPath $installRoot -Recurse -Force
    }
}
if ((Test-Path -LiteralPath $installRoot) -and !(Test-OwnedInstallRoot -InstallRoot $installRoot) -and !(Test-DirectoryEmpty -Path $installRoot)) {
    throw "Install directory already exists and is not empty. Choose an empty directory or a VSR Online install directory: $installRoot"
}
New-Item -ItemType Directory -Force -Path $installRoot | Out-Null
if (!(Test-OwnedInstallRoot -InstallRoot $installRoot)) {
    Write-InstallMarker -InstallRoot $installRoot
}

Write-Step "Copying VSR source files"
Copy-ProjectSource -SourceDir $sourceDir -TargetDir $projectDir

$pythonExe = Install-EmbeddedPython -PythonDir $pythonDir -DownloadDir $downloadDir

if (!(Test-Path -LiteralPath $venvDir)) {
    Write-Step "Creating Python virtual environment"
    & $pythonExe -m venv $venvDir
}
$venvPython = Join-Path $venvDir "Scripts\python.exe"

$resolvedBackend = Resolve-Backend -RequestedBackend $Backend
$gpuName = Get-NvidiaGpuName
if ([string]::IsNullOrWhiteSpace($gpuName)) {
    Write-Host "GPU: No NVIDIA GPU detected"
} else {
    Write-Host "GPU: $gpuName"
}
Write-Host "Selected backend: $resolvedBackend"

Install-PythonPackages -VenvPython $venvPython -ProjectDir $projectDir -ResolvedBackend $resolvedBackend
Write-RunScript -ProjectDir $projectDir
New-DesktopShortcut -ProjectDir $projectDir

Write-Step "Verifying runtime"
& $venvPython -c "import cv2, torch; print('cv2 OK'); print('torch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Write-Host ""
Write-Host "Install completed." -ForegroundColor Green
Write-Host "Install directory: $projectDir"
Write-Host "Start VSR from desktop shortcut or run: $projectDir\run_vsr.bat"
