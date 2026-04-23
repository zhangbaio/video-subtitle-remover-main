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

function Assert-OwnedInstallRoot {
    param([string]$InstallRoot)
    if (!(Test-OwnedInstallRoot -InstallRoot $InstallRoot)) {
        throw "Refusing to delete install directory because it is not marked as a VSR Online installation: $InstallRoot"
    }
}

if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = Join-Path $env:LOCALAPPDATA "Programs\VSR_Online"
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
    Write-Step "Removing install directory"
    Remove-Item -LiteralPath $installRoot -Recurse -Force
} else {
    Write-Host "Install directory does not exist."
}

Write-Host ""
Write-Host "Uninstall completed." -ForegroundColor Green
