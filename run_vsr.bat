@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" gui.py
    exit /b %ERRORLEVEL%
)

echo Cannot find local runtime: "%~dp0.venv\Scripts\python.exe"
echo Please run scripts\install_vsr_online.bat first, or start VSR from the installed shortcut.
pause
exit /b 1
