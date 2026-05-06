# 最新代码打包 ASCII 安装器教程（RTX 30 / RTX 40 / RTX 50）

本文档基于当前这台机器的实际仓库路径、Inno Setup 路径和现有构建环境整理，目标是：

- 使用仓库最新代码重新生成对应的 `Release` 目录
- 再编译出对应的 ASCII 安装器输出目录
- 为 `RTX 30 / RTX 40 / RTX 50` 分别提供完整流程和一键脚本

当前仓库根目录：

```text
D:\code\video-subtitle-remover-main
```

## 1. 三个显卡系列与打包目标的对应关系

| 显卡系列 | CUDA / Torch 目标 | QPT 输出目录 | Inno Setup 模板 | 安装器输出目录 |
| --- | --- | --- | --- | --- |
| RTX 30 系列 | `cu118` | `vsr_out_cu118` | `installer-cu118-ascii.iss` | `installer-dist-cu118-ascii` |
| RTX 40 系列 | `cu126` | `vsr_out_cu126_clean` | `installer-cu126-ascii-clean.iss` | `installer-dist-cu126-ascii` |
| RTX 50 系列 | `cu128` | `vsr_out_cu128_clean` | `installer-cu128-ascii.iss` | `installer-dist-cu128-ascii` |

## 2. 当前本机已确认的环境

这台机器已经确认存在：

- 仓库根目录：`D:\code\video-subtitle-remover-main`
- Inno Setup 编译器：`C:\Program Files (x86)\Inno Setup 6\ISCC.exe`
- QPT 现成环境：`D:\code\video-subtitle-remover-main\.venv-build-cu128`
- 现成 RTX 40/50 相关目录：
  - `vsr_out_cu126`
  - `vsr_out_cu128_clean`
  - `installer-dist-cu128-ascii`

同时也确认到一个很重要的事实：

```text
.\.venv-build-cu128\Scripts\pip.exe freeze
```

当前环境里实际是：

```text
torch==2.7.0+cu126
torchvision==0.22.0+cu126
```

也就是说：

- 目录名叫 `.venv-build-cu128`
- 但里面实际是 **cu126 环境**

所以在这台机器上：

- **RTX 40 脚本大概率可以直接用这个环境跑**
- **RTX 50 不能直接假设它可用**
- **RTX 30 也不能直接假设已准备好**

这也是我新增脚本里专门做 `torch` 版本校验的原因。

## 3. 新增的一键打包脚本

我已经在 `scripts/` 下新增了：

### 通用脚本

```text
scripts\build_installer_ascii.ps1
```

### PowerShell 快捷入口

```text
scripts\build_installer_rtx30.ps1
scripts\build_installer_rtx40.ps1
scripts\build_installer_rtx50.ps1
```

### 双击/命令行一键入口

```text
scripts\build_installer_rtx30.bat
scripts\build_installer_rtx40.bat
scripts\build_installer_rtx50.bat
```

## 4. 这些脚本会做什么

通用脚本 `build_installer_ascii.ps1` 会自动执行这些步骤：

1. 校验仓库路径、`gui.py`、`design\vsr.ico`
2. 选择对应 backend 的构建虚拟环境
3. 读取虚拟环境里的 `torch.__version__`
4. 校验它是否真的是目标 backend
5. 调用 QPT 重新生成对应 `Release`
6. 基于现有 `.iss` 模板生成一个临时安装器脚本
7. 自动把 `.iss` 里的 `SourceDir` 和 `OutputDir` 改成当前实际路径
8. 调用 `ISCC.exe` 编译安装器

这样你就不用手工改 `.iss` 文件里的绝对路径了。

## 5. 通用脚本的主要参数

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_installer_ascii.ps1 -BackendPreset rtx40
```

支持的主要参数：

- `-BackendPreset rtx30|rtx40|rtx50`
- `-VenvPath <路径>`：手工指定构建虚拟环境
- `-RepoRoot <路径>`：手工指定仓库根目录
- `-InnoSetupPath <路径>`：手工指定 `ISCC.exe`
- `-Clean`：先删旧产物再重新打
- `-SkipQpt`：跳过 QPT，只编 installer
- `-SkipInstaller`：只跑 QPT，不编 installer

## 6. 开始前统一建议

### 6.1 先同步到最新代码

```powershell
cd D:\code\video-subtitle-remover-main
git status
git pull --rebase origin master
```

如果你的主分支不是 `master`，改成对应分支即可。

### 6.2 建议清理旧产物

脚本已经支持 `-Clean`，推荐直接用：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_installer_ascii.ps1 -BackendPreset rtx40 -Clean
```

## 7. RTX 30 系列完整教程

目标：

- `torch` 目标后缀：`+cu118`
- 输出目录：`vsr_out_cu118`
- 安装器目录：`installer-dist-cu118-ascii`

### 7.1 准备构建环境

优先使用：

```text
D:\code\video-subtitle-remover-main\.venv-build-cu118
```

如果本机没有，先创建：

```powershell
cd D:\code\video-subtitle-remover-main

C:\Users\PC\miniconda3\python.exe -m venv .venv-build-cu118
.\.venv-build-cu118\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install qpt==1.0b8 QPT-SDK==1.0.1
pip install -r .\requirements.txt
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
```

确认：

```powershell
.\.venv-build-cu118\Scripts\python.exe -c "import torch; print(torch.__version__)"
```

应包含：

```text
+cu118
```

### 7.2 一键打包

```powershell
cd D:\code\video-subtitle-remover-main
powershell -ExecutionPolicy Bypass -File .\scripts\build_installer_rtx30.ps1 -Clean
```

或者：

```cmd
scripts\build_installer_rtx30.bat -Clean
```

### 7.3 产物位置

```text
D:\code\video-subtitle-remover-main\installer-dist-cu118-ascii
```

## 8. RTX 40 系列完整教程

目标：

- `torch` 目标后缀：`+cu126`
- 输出目录：`vsr_out_cu126_clean`
- 安装器目录：`installer-dist-cu126-ascii`

### 8.1 准备构建环境

标准环境名建议是：

```text
D:\code\video-subtitle-remover-main\.venv-build-cu126
```

如果没有，可创建：

```powershell
cd D:\code\video-subtitle-remover-main

C:\Users\PC\miniconda3\python.exe -m venv .venv-build-cu126
.\.venv-build-cu126\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install qpt==1.0b8 QPT-SDK==1.0.1
pip install -r .\requirements.txt
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
```

确认：

```powershell
.\.venv-build-cu126\Scripts\python.exe -c "import torch; print(torch.__version__)"
```

应包含：

```text
+cu126
```

### 8.2 本机特殊说明

当前这台机器虽然没有看到 `.venv-build-cu126` 目录，但现有：

```text
D:\code\video-subtitle-remover-main\.venv-build-cu128
```

里面实际装的是：

```text
torch==2.7.0+cu126
```

所以在这台机器上，如果你想立刻打 RTX 40，可以直接显式指定它：

```powershell
cd D:\code\video-subtitle-remover-main

powershell -ExecutionPolicy Bypass -File .\scripts\build_installer_rtx40.ps1 `
  -VenvPath .\.venv-build-cu128 `
  -Clean
```

### 8.3 一键打包

标准环境名可用时：

```powershell
cd D:\code\video-subtitle-remover-main
powershell -ExecutionPolicy Bypass -File .\scripts\build_installer_rtx40.ps1 -Clean
```

或者：

```cmd
scripts\build_installer_rtx40.bat -Clean
```

### 8.4 产物位置

```text
D:\code\video-subtitle-remover-main\installer-dist-cu126-ascii
```

## 9. RTX 50 系列完整教程

目标：

- `torch` 目标后缀：`+cu128`
- 输出目录：`vsr_out_cu128_clean`
- 安装器目录：`installer-dist-cu128-ascii`

### 9.1 准备构建环境

标准环境名：

```text
D:\code\video-subtitle-remover-main\.venv-build-cu128
```

如果当前环境不是 `+cu128`，就需要重建：

```powershell
cd D:\code\video-subtitle-remover-main

if (Test-Path .\.venv-build-cu128) {
    Rename-Item .\.venv-build-cu128 .venv-build-cu128.backup
}

C:\Users\PC\miniconda3\python.exe -m venv .venv-build-cu128
.\.venv-build-cu128\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install qpt==1.0b8 QPT-SDK==1.0.1
pip install -r .\requirements.txt
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

确认：

```powershell
.\.venv-build-cu128\Scripts\python.exe -c "import torch; print(torch.__version__)"
```

应包含：

```text
+cu128
```

### 9.2 一键打包

```powershell
cd D:\code\video-subtitle-remover-main
powershell -ExecutionPolicy Bypass -File .\scripts\build_installer_rtx50.ps1 -Clean
```

或者：

```cmd
scripts\build_installer_rtx50.bat -Clean
```

### 9.3 产物位置

```text
D:\code\video-subtitle-remover-main\installer-dist-cu128-ascii
```

## 10. 如果你只想编 installer，不想重新跑 QPT

适用于：

- `Release` 已经生成好了
- 只是改了 `.iss`
- 或只是想重出安装器

例如 RTX 50：

```powershell
cd D:\code\video-subtitle-remover-main
powershell -ExecutionPolicy Bypass -File .\scripts\build_installer_rtx50.ps1 -SkipQpt
```

## 11. 如果你只想重跑 QPT，不想编 installer

例如 RTX 40：

```powershell
cd D:\code\video-subtitle-remover-main
powershell -ExecutionPolicy Bypass -File .\scripts\build_installer_rtx40.ps1 -SkipInstaller
```

## 12. 打包完成后的自检

建议至少检查：

```powershell
Get-ChildItem .\installer-dist-cu118-ascii
Get-ChildItem .\installer-dist-cu126-ascii
Get-ChildItem .\installer-dist-cu128-ascii
```

并确认输出里至少有：

- `*.exe`
- `*-1.bin`
- `*-2.bin`

同时建议：

1. 双击安装器，确认向导可启动
2. 安装后确认 `VSR.exe` 可启动
3. 用一段小视频做回归验证，确认安装器对应的是最新代码

## 13. 常见问题

### 13.1 脚本提示 `torch` 后缀不匹配

这是正常保护逻辑。说明你当前虚拟环境不是目标 backend。

例如：

- 打 `RTX 50`
- 但虚拟环境里实际是 `torch==2.7.0+cu126`

脚本会直接中止，避免误把 `cu126` 环境打成 `cu128` 安装器。

### 13.2 本机现在为什么 RTX 50 不能直接一键打

因为当前机器上已确认：

```text
.venv-build-cu128 实际是 torch==2.7.0+cu126
```

所以它更接近 RTX 40 构建环境，而不是 RTX 50。

### 13.3 `ISCC.exe` 路径不对

当前本机已确认路径是：

```text
C:\Program Files (x86)\Inno Setup 6\ISCC.exe
```

如果你后续迁移到别的机器，可以用：

```powershell
-InnoSetupPath '你的ISCC.exe路径'
```

### 13.4 `installer-cu126-ascii-clean.iss` 和 `installer-cu126-ascii.iss` 有什么关系

当前一键脚本默认使用：

```text
installer-cu126-ascii-clean.iss
```

因为它和 `vsr_out_cu126_clean` / `installer-dist-cu126-ascii` 的命名更一致。

而且脚本会自动生成一份临时 `.iss`，并把 `SourceDir` / `OutputDir` 改成当前实际路径，所以你不需要手改模板。

