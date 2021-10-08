git clone https://gitee.com/jackboosy/vcpkg.git --depth=1
set X_VCPKG_ASSET_SOURCES=x-azurl,http://106.15.181.5/
cd vcpkg
bootstrap-vcpkg.bat
vcpkg integrate install
vcpkg install openvdb:x64-windows
vcpkg install eigen3:x64-windows
vcpkg install cgal:x64-windows
vcpkg install openblas:x64-windows
vcpkg install lapack:x64-windows
cd ..
git clone https://gitee.com/zenustech/zeno.git
cd zeno
cmake -B build -DCMAKE_TOOLCHAIN_FILE=%CD%/../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --parallel 8
