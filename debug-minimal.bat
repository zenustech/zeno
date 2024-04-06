cmake -B Debug ^
-DCMAKE_BUILD_TYPE=Debug ^
-DCMAKE_PREFIX_PATH=C:/Qt/lib/cmake ^
-DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake" ^
-DZENO_WITH_PYTHON3=OFF ^
-DZENO_WITH_ZenoFX:BOOL=ON ^
-DZENO_WITH_zenvdb:BOOL=ON ^
-DZENO_ENABLE_OPTIX:BOOL=OFF