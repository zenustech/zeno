[CmdletBinding()]
param(
    [ValidateSet("minimum","full")]
    [string]$Preset = "minimum",

    [ValidateSet("Debug","Release")]
    [string]$Build = "Release",

    [ValidateSet("2019","2022","Latest")]
    [string]$VSVersion = "Latest"
)

$ErrorActionPreference = "Stop"

# ------------------------------------------------------------
# Logging Helpers
# ------------------------------------------------------------

function Write-Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "[ OK ]  $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Fail($msg)  { Write-Host "[FAIL]  $msg" -ForegroundColor Red }

# ------------------------------------------------------------
# Validate Required Tools
# ------------------------------------------------------------

function Require-Command($name) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Fail "$name not found in PATH."
        exit 1
    }
    $source = $cmd.Source
    Write-Ok "$name found at: $source"
}

Write-Info "Checking required tools..."
Require-Command("cmake")
Require-Command("ninja")

# ------------------------------------------------------------
# Validate VCPKG
# ------------------------------------------------------------

if ($null -eq $env:VCPKG_ROOT) { 
    Write-Info "Environment variables env:VCPKG_ROOT not found"

    Write-Info "Please set you vcpkg path as VCPKG_ROOT."
    $env:VCPKG_ROOT = Read-Host 'VCPKG_ROOT'
    # return;
}

if (-not (Test-Path "$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake")) {
    Write-Fail "Invalid VCPKG_ROOT path."
    exit 1
}

Write-Ok "Using vcpkg at $env:VCPKG_ROOT"

# ------------------------------------------------------------
# Locate Visual Studio (2019, 2022, 2026+)
# ------------------------------------------------------------
# $env:VS_PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community"

if ($null -eq $env:VS_PATH) {

    Write-Output "VS_PATH not set. Trying automatic detection..."

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

    if (!(Test-Path $vswhere)) {
        Write-Error "vswhere.exe not found. Please install Visual Studio."
        return
    }

    # Get latest installed VS with C++ tools
    $vsPath = & $vswhere `
        -latest `
        -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath

    if ([string]::IsNullOrEmpty($vsPath)) {
        Write-Error "No suitable Visual Studio installation found."
        return
    }

    $env:VS_PATH = $vsPath
    Write-Output "Using Visual Studio at: $env:VS_PATH"
}

# ------------------------------------------------------------
# Enter VS Dev Shell
# ------------------------------------------------------------

Import-Module ("${env:VS_PATH}\Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath "${env:VS_PATH}" -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"

$shell_name = (Get-Process -Id $PID).name
Write-Host $shell_name

# if ("pwsh" -eq $shell_name) {
#     cmd /K '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && pwsh.exe'
# } else {
#     cmd /K '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && powershell.exe'
# }

Write-Info "Configuring project..."
Write-Info "Preset: $Preset"
Write-Info "Config : $Build"

$p = $Preset
$b = $Build

$commonArgs = @(
    "-G", "Ninja", "-B", "build", "-DCMAKE_BUILD_TYPE=${b}",
    "-DCMAKE_CUDA_FLAGS='-allow-unsupported-compiler'",
    "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",

    "-DDEACTIVATE_ZLIB=ON",
    "-DZENO_WITH_zenvdb:BOOL=ON",
    "-DZENO_SYSTEM_OPENVDB:BOOL=OFF",
    "-DZENO_WITH_ZenoFX:BOOL=ON",
    "-DZENO_ENABLE_OPTIX:BOOL=ON",
    "-DZENO_WITH_FBX:BOOL=ON",
    "-DZENO_WITH_Alembic:BOOL=ON",
    "-DZENO_WITH_MeshSubdiv:BOOL=ON",
    "-DCMAKE_POLICY_VERSION_MINIMUM='3.5'",
    "-DOTK_USE_VCPKG:BOOL=OFF",
    "-DOTK_FETCH_CONTENT:BOOL=OFF",
    "-DOTK_BUILD_DOCS:BOOL=OFF",
    "-DOTK_BUILD_TESTS:BOOL=OFF",
    "-DOTK_BUILD_EXAMPLES:BOOL=OFF",
    "-DOTK_LIBRARIES=OmmBaking"
)

if ($p.ToLower() -ne "full") {
    Write-Info "Making minimum build with Optix..."
} else {
    Write-Info "Making full build..."

$extraArgs = @(
    "-DZENO_WITH_CUDA:BOOL=ON"
    "-DZENOFX_ENABLE_OPENVDB:BOOL=ON"
    "-DZENOFX_ENABLE_LBVH:BOOL=ON"
    "-DZENO_WITH_FastFLIP:BOOL=ON"
    "-DZENO_WITH_FEM:BOOL=ON"
    "-DZENO_WITH_Rigid:BOOL=ON"
    "-DZENO_WITH_cgmesh:BOOL=ON"
    "-DZENO_WITH_oldzenbase:BOOL=ON"
    "-DZENO_WITH_TreeSketch:BOOL=ON"
    "-DZENO_WITH_Skinning:BOOL=ON"
    "-DZENO_WITH_Euler:BOOL=ON"
    "-DZENO_WITH_Functional:BOOL=ON"
    "-DZENO_WITH_LSystem:BOOL=ON"
    "-DZENO_WITH_mesher:BOOL=ON"
    "-DZENO_WITH_DemBones:BOOL=ON"
    "-DZENO_WITH_SampleModel:BOOL=ON"
    "-DZENO_WITH_CalcGeometryUV:BOOL=ON"
    "-DZENO_WITH_Audio:BOOL=ON"
    "-DZENO_WITH_PBD:BOOL=ON"
    "-DZENO_WITH_GUI:BOOL=ON"
    "-DZENO_WITH_ImgCV:BOOL=ON"
    "-DZENO_WITH_TOOL_FLIPtools:BOOL=ON"
    "-DZENO_WITH_TOOL_cgmeshTools:BOOL=ON"
    "-DZENO_WITH_TOOL_BulletTools:BOOL=ON"
    "-DZENO_WITH_TOOL_HerculesTools:BOOL=ON"
)
}

cmake @commonArgs @extraArgs

Write-Info "Building..."

cmake --build build --config ${b}
cp ./build/compile_commands.json ./