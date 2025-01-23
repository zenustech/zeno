param( [string]$p, [string]$b)

if (Get-Command "cmake" -errorAction SilentlyContinue)
{
    Write-Output "Found cmake"
} else  {
    Write-Output "Please install cmake"
    return
}

if (Get-Command "ninja" -errorAction SilentlyContinue)
{
    Write-Output "Found ninja"
} else {
    Write-Output "Please install ninja"
    return
}

if ($null -eq $env:VCPKG_ROOT) { 
    Write-Output "Environment variables env:VCPKG_ROOT not found"

    Write-Output "Please set you vcpkg path as VCPKG_ROOT."
    $env:VCPKG_ROOT = Read-Host 'VCPKG_ROOT'
    # return;
}

$DEFAULT_VS_PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community"

if ($null -eq $env:VS_PATH) { 

    Write-Output "Environment variables env:VS_PATH not found"

    if ([System.IO.Directory]::Exists($DEFAULT_VS_PATH)) {

        Write-Output "Trying to use default Visual studio Path"
        $env:VS_PATH = $DEFAULT_VS_PATH
    } else {
    
        Write-Output "Please set you Visual Studio install Path as VS_PATH, for example below:"
        Write-Output '$Env:VS_PATH="C:\Program Files\Microsoft Visual Studio\2022\Community"'

        $env:VS_PATH = Read-Host 'VS_PATH'
        # return;
    }
} 

Import-Module ("$env:VS_PATH\\Common7\\Tools\\Microsoft.VisualStudio.DevShell.dll");
Enter-VsDevShell -VsInstallPath "${env:VS_PATH}" -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"

$shell_name = (Get-Process -Id $PID).name
Write-Host $shell_name

# if ("pwsh" -eq $shell_name) {
#     cmd /K '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && pwsh.exe'
# } else {
#     cmd /K '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && powershell.exe'
# }

if ($b.ToLower() -ne "debug") {
    $b = "Release"
} else {
    $b = "Debug"
}

Write-Host $p
Write-Host $b

if ($p.ToLower() -ne "full") {

Write-Output "Making minimum build with Optix..."

cmake -G Ninja -B build -DCMAKE_BUILD_TYPE="${b}" `
    -DCMAKE_TOOLCHAIN_FILE="${env:VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" `
    -DDEACTIVATE_ZLIB=ON `
    -DZENO_WITH_zenvdb:BOOL=ON `
    -DZENO_SYSTEM_OPENVDB:BOOL=OFF `
    -DZENO_WITH_ZenoFX:BOOL=ON `
    -DZENO_ENABLE_OPTIX:BOOL=ON `
    -DZENO_WITH_FBX:BOOL=ON `
    -DZENO_WITH_Alembic:BOOL=ON `
    -DZENO_WITH_MeshSubdiv:BOOL=ON `
    -DZENO_WITH_CalcGeometryUV:BOOL=ON `
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

} else {

Write-Output "Making full build..."

cmake -G Ninja -B build -DCMAKE_BUILD_TYPE="${b}" `
    -DCMAKE_TOOLCHAIN_FILE="${env:VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" `
    -DDEACTIVATE_ZLIB=ON `
    -DZENO_WITH_CUDA:BOOL=ON `
    -DZENO_WITH_ZenoFX:BOOL=ON `
    -DZENO_ENABLE_OPTIX:BOOL=ON `
    -DZENO_SYSTEM_OPENVDB:BOOL=OFF `
    -DZENOFX_ENABLE_OPENVDB:BOOL=ON `
    -DZENOFX_ENABLE_LBVH:BOOL=ON `
    -DZENO_WITH_zenvdb:BOOL=ON `
    -DZENO_WITH_FastFLIP:BOOL=ON `
    -DZENO_WITH_FEM:BOOL=ON `
    -DZENO_WITH_Rigid:BOOL=ON `
    -DZENO_WITH_cgmesh:BOOL=ON `
    -DZENO_WITH_oldzenbase:BOOL=ON `
    -DZENO_WITH_TreeSketch:BOOL=ON `
    -DZENO_WITH_Skinning:BOOL=ON `
    -DZENO_WITH_Euler:BOOL=ON `
    -DZENO_WITH_Functional:BOOL=ON `
    -DZENO_WITH_LSystem:BOOL=ON `
    -DZENO_WITH_mesher:BOOL=ON `
    -DZENO_WITH_Alembic:BOOL=ON `
    -DZENO_WITH_FBX:BOOL=ON `
    -DZENO_WITH_DemBones:BOOL=ON `
    -DZENO_WITH_SampleModel:BOOL=ON `
    -DZENO_WITH_CalcGeometryUV:BOOL=ON `
    -DZENO_WITH_MeshSubdiv:BOOL=ON `
    -DZENO_WITH_Audio:BOOL=ON `
    -DZENO_WITH_PBD:BOOL=ON `
    -DZENO_WITH_GUI:BOOL=ON `
    -DZENO_WITH_ImgCV:BOOL=ON `
    -DZENO_WITH_TOOL_FLIPtools:BOOL=ON `
    -DZENO_WITH_TOOL_cgmeshTools:BOOL=ON `
    -DZENO_WITH_TOOL_BulletTools:BOOL=ON `
    -DZENO_WITH_TOOL_HerculesTools:BOOL=ON `
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
}

cmake --build build --config Release
cp ./build/compile_commands.json ./