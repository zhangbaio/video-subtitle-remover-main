@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0build_installer_ascii.ps1" -BackendPreset rtx40 %*
exit /b %ERRORLEVEL%
