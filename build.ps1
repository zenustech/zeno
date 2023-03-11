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
    Write-Output "Please set you vcpkg path as VCPKG_ROOT."
    return;
}

if ($null -eq $env:VS_PATH) { 
    Write-Output "Please set you Visual Studio install Path as VS_PATH, for example below:"
    Write-Output '$Env:VS_PATH="C:\Program Files\Microsoft Visual Studio\2022\Community"'
    return;
} 

Import-Module ("$env:VS_PATH\\Common7\\Tools\\Microsoft.VisualStudio.DevShell.dll");
Enter-VsDevShell -VsInstallPath "${env:VS_PATH}" -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"

$shell_name = (Get-Process -Id $PID).name
echo $shell_name

# if ("pwsh" -eq $shell_name) {
#     cmd /K '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && pwsh.exe'
# } else {
#     cmd /K '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && powershell.exe'
# }

cmake -G Ninja -B ninja -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_TOOLCHAIN_FILE="${env:VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" `
    -DZENO_WITH_zenvdb:BOOL=ON `
    -DZENO_SYSTEM_OPENVDB=OFF `
    -DZENO_WITH_ZenoFX:BOOL=ON `
    -DZENO_ENABLE_OPTIX:BOOL=ON `
    -DZENO_WITH_FBX:BOOL=ON `
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# cmake -G Ninja -B ninja -DCMAKE_BUILD_TYPE=Release `
#     -DCMAKE_TOOLCHAIN_FILE="${env:VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" `
#     -DZENO_WITH_ZenoFX:BOOL=ON `
#     -DZENOFX_ENABLE_OPENVDB:BOOL=ON `
#     -DZENOFX_ENABLE_LBVH:BOOL=ON `
#     -DZENO_WITH_zenvdb:BOOL=ON `
#     -DZENO_WITH_FastFLIP:BOOL=ON `
#     -DZENO_WITH_FEM:BOOL=ON `
#     -DZENO_WITH_Rigid:BOOL=ON `
#     -DZENO_WITH_cgmesh:BOOL=ON `
#     -DZENO_WITH_oldzenbase:BOOL=ON `
#     -DZENO_WITH_TreeSketch:BOOL=ON `
#     -DZENO_WITH_Skinning:BOOL=ON `
#     -DZENO_WITH_Euler:BOOL=ON `
#     -DZENO_WITH_Functional:BOOL=ON `
#     -DZENO_WITH_LSystem:BOOL=ON `
#     -DZENO_WITH_mesher:BOOL=ON `
#     -DZENO_WITH_Alembic:BOOL=ON `
#     -DZENO_WITH_FBX:BOOL=ON `
#     -DZENO_WITH_DemBones:BOOL=ON `
#     -DZENO_WITH_SampleModel:BOOL=ON `
#     -DZENO_WITH_CalcGeometryUV:BOOL=ON `
#     -DZENO_WITH_MeshSubdiv:BOOL=ON `
#     -DZENO_WITH_Audio:BOOL=ON `
#     -DZENO_WITH_PBD:BOOL=ON `
#     -DZENO_WITH_GUI:BOOL=ON `
#     -DZENO_WITH_ImgCV:BOOL=ON

cmake --build ninja --config Release