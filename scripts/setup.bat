git clone https://github.com/microsoft/vcpkg.git --depth=1
cd vcpkg
bootstrap-vcpkg.bat
vcpkg integrate install
vcpkg install openvdb:x64-windows
vcpkg install eigen3:x64-windows
vcpkg install cgal:x64-windows
vcpkg install openblas:x64-windows
vcpkg install lapack:x64-windows
cd ..
git clone https://github.com/zenustech/zeno.git
cd zeno
cmake -B build -DCMAKE_TOOLCHAIN_FILE=%CD%/../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --parallel 8
