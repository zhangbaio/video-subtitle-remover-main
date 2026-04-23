@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%uninstall_vsr_online.ps1"

if /I "%~1"=="/?" goto :help
if /I "%~1"=="-?" goto :help
if /I "%~1"=="--help" goto :help
if /I "%~1"=="help" goto :help

if not exist "%PS_SCRIPT%" (
    echo Cannot find uninstall_vsr_online.ps1
    echo Expected: "%PS_SCRIPT%"
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo Uninstall completed successfully.
) else (
    echo Uninstall failed. Exit code: %EXIT_CODE%
)
pause

exit /b %EXIT_CODE%

:help
echo VSR online uninstaller
echo.
echo Usage:
echo   uninstall_vsr_online.bat
echo   uninstall_vsr_online.bat -Yes
echo   uninstall_vsr_online.bat -InstallDir "D:\Apps\VSR_Online" -Yes
echo   uninstall_vsr_online.bat -KeepDownloads
echo.
echo If -InstallDir is omitted, the PowerShell uninstaller prompts for an install directory.
echo.
pause
exit /b 0
