; 脚本由 Inno Setup 脚本向导 生成！
; 有关创建 Inno Setup 脚本文件的详细资料请查阅帮助文档！

#define MyAppName "zeno2"
#define MyAppVersion "1.0.0.0"
#define MyAppPublisher "深圳泽森科技有限公司"
#define MyAppURL "https://zenustech.com/"
#define MyAppExeName "zenoedit.exe"
#define MyAppAssocName "zeno graph"
#define MyAppAssocExt ".zsg"
#define MyAppAssocKey StringChange(MyAppAssocName, " ", "") + MyAppAssocExt

[Setup]
; 注: AppId的值为单独标识该应用程序。
; 不要为其他安装程序使用相同的AppId值。
; (若要生成新的 GUID，可在菜单中点击 "工具|生成 GUID"。)
AppId={{F4DE174F-DBA8-4737-AFC4-E8ACB7CD8FA5}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
ChangesAssociations=yes
DisableProgramGroupPage=yes
; 以下行取消注释，以在非管理安装模式下运行（仅为当前用户安装）。
;PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=C:\Users\pgjgg\Desktop\work\202302\zeno\install_outdir
OutputBaseFilename=zeno2_setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\Alembic.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\assimp-vc143-mt.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\assimp-vc143-mt.dll.manifest"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\assimp-vc143-mt.exp"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\assimp-vc143-mt.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\brotlicommon.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\brotlidec.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\Bullet3Common.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\BulletCollision.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\BulletDynamics.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\BulletInverseDynamics.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\BulletSoftBody.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\BussIK.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\bz2.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\cudart64_12.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\cufft64_11.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\cufftw64_11.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\freetype.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\HACD.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\harfbuzz.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\hdf5.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\icudt71.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\icuin71.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\icuuc71.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\Iex-3_1.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\IlmThread-3_1.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\Imath-3_2.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\IrrXML.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\jpeg62.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\libblosc.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\libomp140.x86_64.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\libopenvdb.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\libpng16.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\LinearMath.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\nvrtc64_120_0.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\nvrtc-builtins64_120.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\OpenEXR-3_1.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\osdCPU.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\partio.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\pcre2-16.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\qt.conf"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\Qt5Core.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\Qt5Gui.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\Qt5Network.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\Qt5OpenGL.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\Qt5Svg.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\Qt5Widgets.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\tbb.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\URDFImporter.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\VHACD.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\xatlasUVCore.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\zeno.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\zeno.dll.manifest"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\zeno.exp"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\zeno.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\zenoedit.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\zenoedit.exe.manifest"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\ZFX.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\zlib1.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\zpc.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\zpccore.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\zpccuda.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\zpcomp.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\zpctool.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\zspartio.lib"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\zstd.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\pgjgg\Desktop\work\202302\zeno\zeno-2023-0215-122452-nt\bin\plugins\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; 注意: 不要在任何共享系统文件上使用“Flags: ignoreversion”

[Registry]
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocKey}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}"; ValueType: string; ValueName: ""; ValueData: "{#MyAppAssocName}"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""
Root: HKA; Subkey: "Software\Classes\Applications\{#MyAppExeName}\SupportedTypes"; ValueType: string; ValueName: ".myp"; ValueData: ""

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

