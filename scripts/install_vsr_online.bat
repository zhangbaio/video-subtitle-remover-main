@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%install_vsr_online.ps1"

if /I "%~1"=="/?" goto :help
if /I "%~1"=="-?" goto :help
if /I "%~1"=="--help" goto :help
if /I "%~1"=="help" goto :help

if not exist "%PS_SCRIPT%" (
    echo Cannot find install_vsr_online.ps1
    echo Expected: "%PS_SCRIPT%"
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Install failed. Exit code: %EXIT_CODE%
    pause
)

exit /b %EXIT_CODE%

:help
echo VSR online installer
echo.
echo Usage:
echo   install_vsr_online.bat
echo   install_vsr_online.bat -Force
echo   install_vsr_online.bat -Backend cu126
echo   install_vsr_online.bat -InstallDir "D:\Apps\VSR_Online"
echo.
echo Backends:
echo   auto, cpu, directml, cu118, cu126, cu128
echo.
pause
exit /b 0
