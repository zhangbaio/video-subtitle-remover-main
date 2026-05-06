@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0build_installer_ascii.ps1" -BackendPreset rtx30 %*
exit /b %ERRORLEVEL%
