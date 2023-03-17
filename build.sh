# export CC=/usr/bin/clang
# export CXX=/usr/bin/clang++

cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DZENO_WITH_zenvdb:BOOL=ON \
    -DZENO_SYSTEM_OPENVDB=OFF \
    -DZENO_WITH_ZenoFX:BOOL=ON \
    -DZENO_ENABLE_OPTIX:BOOL=ON \
    -DZENO_WITH_FBX:BOOL=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
# -DCMAKE_TOOLCHAIN_FILE="${env:VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" \

cmake --build build --parallel $(nproc) \

ln -s ./build/compile_commands.json ./