param(
    [string]$InstallDir = "",
    [switch]$KeepDownloads,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"

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

function Resolve-DefaultInstallDir {
    $scriptRoot = Resolve-ScriptRoot
    $sourceDir = Split-Path -Parent $scriptRoot
    $parentDir = Split-Path -Parent $sourceDir
    return (Join-Path $parentDir "VSR_Online")
}

function Get-Text {
    param([int[]]$CodePoints)
    return -join ($CodePoints | ForEach-Object { [char]$_ })
}

function Read-UninstallDir {
    param([string]$DefaultInstallDir)
    $titleText = Get-Text @(0x8BF7, 0x9009, 0x62E9, 0x8981, 0x5378, 0x8F7D, 0x7684, 0x5B89, 0x88C5, 0x76EE, 0x5F55)
    $defaultText = Get-Text @(0x9ED8, 0x8BA4, 0x76EE, 0x5F55)
    $promptText = Get-Text @(0x76F4, 0x63A5, 0x56DE, 0x8F66, 0x4F7F, 0x7528, 0x9ED8, 0x8BA4, 0x76EE, 0x5F55, 0xFF0C, 0x6216, 0x8F93, 0x5165, 0x81EA, 0x5B9A, 0x4E49, 0x5B89, 0x88C5, 0x76EE, 0x5F55)
    Write-Host ""
    Write-Host $titleText
    Write-Host "${defaultText}: $DefaultInstallDir"
    $inputDir = Read-Host $promptText
    if ([string]::IsNullOrWhiteSpace($inputDir)) {
        return $DefaultInstallDir
    }
    return $inputDir.Trim('"')
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

function Test-LegacyInstallRoot {
    param([string]$InstallRoot)
    $projectDir = Join-Path $InstallRoot "video-subtitle-remover-main"
    $pythonExe = Join-Path $InstallRoot "Python310\python.exe"
    return (
        (Test-Path -LiteralPath (Join-Path $projectDir "gui.py")) -or
        (Test-Path -LiteralPath (Join-Path $projectDir "run_vsr.bat")) -or
        (Test-Path -LiteralPath $pythonExe)
    )
}

function Assert-OwnedInstallRoot {
    param([string]$InstallRoot)
    if (!(Test-OwnedInstallRoot -InstallRoot $InstallRoot) -and !(Test-LegacyInstallRoot -InstallRoot $InstallRoot)) {
        throw "Refusing to delete install directory because it is not marked as a VSR Online installation: $InstallRoot"
    }
}

function Stop-InstallRootProcesses {
    param([string]$InstallRoot)
    if (!(Test-Path -LiteralPath $InstallRoot)) {
        return
    }

    $fullRoot = [System.IO.Path]::GetFullPath($InstallRoot).TrimEnd("\") + "\"
    $ownProcessId = $PID
    $processes = Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $ownProcessId -and
        $_.ExecutablePath -and
        ([System.IO.Path]::GetFullPath($_.ExecutablePath).StartsWith($fullRoot, [System.StringComparison]::OrdinalIgnoreCase))
    }

    if (!$processes) {
        return
    }

    Write-Step "Stopping VSR processes"
    foreach ($process in $processes) {
        Write-Host "Stopping PID $($process.ProcessId): $($process.Name)"
        try {
            Stop-Process -Id $process.ProcessId -Force -ErrorAction Stop
        } catch {
            Write-Host "Unable to stop PID $($process.ProcessId): $($_.Exception.Message)"
        }
    }
    Start-Sleep -Seconds 2
}

function Remove-ItemWithRetry {
    param(
        [string]$Path,
        [int]$RetryCount = 5
    )
    for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
        try {
            Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
            return
        } catch {
            if ($attempt -ge $RetryCount) {
                throw "Failed to remove $Path. Close any running VSR, Python, terminal, or Explorer windows using this directory, then run uninstall again. Original error: $($_.Exception.Message)"
            }
            Write-Host "Remove failed, retrying ($attempt/$RetryCount): $($_.Exception.Message)"
            Start-Sleep -Seconds 2
            Stop-InstallRootProcesses -InstallRoot $Path
        }
    }
}

if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = Read-UninstallDir -DefaultInstallDir (Resolve-DefaultInstallDir)
}

$installRoot = [System.IO.Path]::GetFullPath($InstallDir)
$desktopShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "VSR.lnk"
$startMenuShortcut = Join-Path ([Environment]::GetFolderPath("Programs")) "VSR.lnk"
$downloadDir = Join-Path $installRoot "downloads"

Assert-SafeInstallRootPath -InstallRoot $installRoot
if (Test-Path -LiteralPath $installRoot) {
    Assert-OwnedInstallRoot -InstallRoot $installRoot
}

Write-Host "This will remove the VSR online installation:"
Write-Host "  $installRoot"
Write-Host ""
Write-Host "It will also remove shortcuts created by install_vsr_online.ps1."
Write-Host "System Python and other applications will not be touched."

if (-not $Yes) {
    $answer = Read-Host "Continue? Type YES to uninstall"
    if ($answer -ne "YES") {
        Write-Host "Canceled."
        exit 0
    }
}

if (Test-Path -LiteralPath $desktopShortcut) {
    Write-Step "Removing desktop shortcut"
    Remove-Item -LiteralPath $desktopShortcut -Force
}

if (Test-Path -LiteralPath $startMenuShortcut) {
    Write-Step "Removing Start Menu shortcut"
    Remove-Item -LiteralPath $startMenuShortcut -Force
}

if ($KeepDownloads -and (Test-Path -LiteralPath $downloadDir)) {
    $backupDir = Join-Path ([System.IO.Path]::GetTempPath()) ("VSR_Online_downloads_" + (Get-Date -Format "yyyyMMddHHmmss"))
    Write-Step "Preserving downloads at $backupDir"
    Move-Item -LiteralPath $downloadDir -Destination $backupDir -Force
}

if (Test-Path -LiteralPath $installRoot) {
    Stop-InstallRootProcesses -InstallRoot $installRoot
    Write-Step "Removing install directory"
    Remove-ItemWithRetry -Path $installRoot
} else {
    Write-Host "Install directory does not exist."
}

Write-Host ""
Write-Host "Uninstall completed." -ForegroundColor Green
