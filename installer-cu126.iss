#define MyAppName "视频字幕去除器 NVIDIA CUDA 12.6"
#define MyAppVersion "1.4.0"
#define MyAppPublisher "YaoFANGUK"
#define MyAppExeName "启动程序.exe"
#define SourceDir "C:\Users\PC\Desktop\zhangbiao\code\vsr_out\Release"

[Setup]
AppId={{E56F4D3F-4A93-46A4-9C7E-CU126VSR140}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\视频字幕去除器 NVIDIA CUDA 12.6
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=C:\Users\PC\Desktop\zhangbiao\code\video-subtitle-remover-main\installer-dist
OutputBaseFilename=VSR_v1.4.0_windows_x64_nvidia_cuda12.6_Setup
SetupIconFile={#SourceDir}\resources\design\vsr.ico
Compression=lzma2/ultra64
SolidCompression=yes
DiskSpanning=yes
DiskSliceSize=2000000000
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务："; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

