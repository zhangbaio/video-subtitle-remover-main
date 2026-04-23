# VSR Online Installer

This folder contains an online installer script for Windows users who do not have Python installed.

## What It Does

`install_vsr_online.ps1` installs a private Python runtime under the user's profile, copies the VSR source code, creates a virtual environment, detects the GPU, installs matching dependencies, and creates a desktop shortcut.

Default install path:

```text
..\VSR_Online
```

鏈彁渚?`-InstallDir` 鏃讹紝瀹夎鍣ㄤ細鎻愮ず杈撳叆瀹夎鐩綍銆?鐩存帴鍥炶溅浼氫娇鐢ㄤ笅杞芥簮鐮佺洰褰曟梺杈圭殑 `VSR_Online` 鏂囦欢澶广€?
## Backend Selection

The default backend is `auto`.

```text
RTX 20 / RTX 30 -> cu118
RTX 40          -> cu126
RTX 50          -> cu128
No NVIDIA GPU   -> cpu
```

You can override it:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_vsr_online.ps1 -Backend cu126
powershell -ExecutionPolicy Bypass -File .\scripts\install_vsr_online.ps1 -Backend cu118
powershell -ExecutionPolicy Bypass -File .\scripts\install_vsr_online.ps1 -Backend cpu
```

## Usage

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_vsr_online.ps1
```

Or double-click:

```text
scripts\install_vsr_online.bat
```

Force a clean reinstall:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_vsr_online.ps1 -Force
```

Using the batch wrapper:

```cmd
scripts\install_vsr_online.bat -Force
```

Install to a custom directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_vsr_online.ps1 -InstallDir "D:\Apps\VSR_Online"
```

## Distribution Notes

Distribute the project folder or a zip of the project folder. The user only needs to run the script. No system Python is required.

The first install requires internet access and can download several GB for NVIDIA CUDA backends.

## Uninstall And Cleanup

To fully remove the online installation, run:

```text
scripts\uninstall_vsr_online.bat
```

Silent confirmation:

```cmd
scripts\uninstall_vsr_online.bat -Yes
```

Custom install directory:

```cmd
scripts\uninstall_vsr_online.bat -InstallDir "D:\Apps\VSR_Online" -Yes
```

This removes the private Python runtime, virtual environment, copied VSR source, downloaded installers under the install directory, and the desktop shortcut. It does not touch system Python or other applications.
