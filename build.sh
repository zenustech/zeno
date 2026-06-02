#!/usr/bin/env bash

# Stop execution if any command fails
set -e

# ------------------------------------------------------------
# Parameter Defaults & Parsing
# ------------------------------------------------------------
PRESET="mini"
BUILD="Release"

print_usage() {
    echo "Usage: $0 [--preset mini|full] [--build Debug|Release]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --build)
            BUILD="$2"
            shift 2
            ;;
        *)
            print_usage
            ;;
    esac
done

# Validate inputs
if [[ "$PRESET" != "mini" && "$PRESET" != "full" ]]; then
    echo "[FAIL] Invalid preset. Use 'mini' or 'full'."
    exit 1
fi

if [[ "$BUILD" != "Debug" && "$BUILD" != "Release" ]]; then
    echo "[FAIL] Invalid build type. Use 'Debug' or 'Release'."
    exit 1
fi

# ------------------------------------------------------------
# Logging Helpers
# ------------------------------------------------------------
write_ok()   { echo -e "\e[32m[ OK ]  $1\e[0m"; }
write_info() { echo -e "\e[36m[INFO]  $1\e[0m"; }
write_fail() { echo -e "\e[31m[FAIL]  $1\e[0m"; }

# ------------------------------------------------------------
# Validate Required Tools & Enforce GCC
# ------------------------------------------------------------
require_command() {
    if ! command -v "$1" &> /dev/null; then
        write_fail "$1 not found in PATH."
        exit 1
    fi
    write_ok "$1 found at: $(command -v "$1")"
}

write_info "Checking required tools..."
require_command "cmake"
require_command "ninja"
require_command "g++"
require_command "gcc"

# Force CMake to use GCC
export CC=$(command -v gcc)
export CXX=$(command -v g++)

export FFMPEG_DIR="./ffmpeg-cmake"

# ------------------------------------------------------------
# Configure Project
# ------------------------------------------------------------
write_info "Configuring project..."
write_info "Preset: $PRESET"
write_info "Config : $BUILD"

# Base CMake Arguments
CMAKE_ARGS=(
    "-G" "Ninja"
    "-B" "build"
    "-DCMAKE_BUILD_TYPE=${BUILD}"
    "-DCMAKE_CXX_FLAGS=-mavx2"
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    "-DDEACTIVATE_ZLIB=ON"
    "-DZENO_WITH_zenvdb:BOOL=ON"
    "-DZENO_SYSTEM_OPENVDB:BOOL=OFF"
    "-DZENO_WITH_ZenoFX:BOOL=ON"
    "-DZENO_ENABLE_OPTIX:BOOL=ON"
    "-DZENO_WITH_FBX:BOOL=ON"
    "-DZENO_WITH_Alembic:BOOL=ON"
    "-DZENO_WITH_MeshSubdiv:BOOL=ON"
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
    "-DOTK_USE_VCPKG:BOOL=OFF"
    "-DOTK_FETCH_CONTENT:BOOL=OFF"
    "-DOTK_BUILD_DOCS:BOOL=OFF"
    "-DOTK_BUILD_TESTS:BOOL=OFF"
    "-DOTK_BUILD_EXAMPLES:BOOL=OFF"
    "-DOTK_LIBRARIES=OmmBaking"
)

# Append extra arguments based on preset selection
if [ "$PRESET" != "full" ]; then
    write_info "Making mini build with Optix..."
else
    write_info "Making full build..."
    CMAKE_ARGS+=(
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
fi

# Execute CMake generation
cmake "${CMAKE_ARGS[@]}"

# ------------------------------------------------------------
# Build Project
# ------------------------------------------------------------
write_info "Building..."
cmake --build build --config "${BUILD}"

# Copy the compile commands to the root directory
cp ./build/compile_commands.json ./
write_ok "Build complete!"
