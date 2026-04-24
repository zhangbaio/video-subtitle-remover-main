#define MyAppName "VSR NVIDIA CUDA 11.8"
#define MyAppVersion "1.4.0"
#define MyAppPublisher "YaoFANGUK"
#define MyAppExeName "VSR.exe"
#define SourceDir "D:\code\video-subtitle-remover-main\vsr_out_cu118\Release"

[Setup]
AppId={{E56F4D3F-4A93-46A4-9C7E-CU118VSR140ASCII}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\VSR_NVIDIA_CUDA_11_8
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=D:\code\video-subtitle-remover-main\installer-dist-cu118-ascii
OutputBaseFilename=VSR_v1.4.0_windows_x64_nvidia_cuda11.8_ASCII_Setup
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
Name: "desktopicon"; Description: "Create desktop shortcut"; GroupDescription: "Additional tasks:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
