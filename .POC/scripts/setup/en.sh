#!/bin/bash
set -e

git clone https://github.com/microsoft/vcpkg.git --depth=1
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
./vcpkg install openvdb:x64-linux
./vcpkg install eigen3:x64-linux
./vcpkg install cgal:x64-linux
./vcpkg install openblas:x64-linux
./vcpkg install lapack:x64-linux
cd ..
git clone https://github.com/zenustech/zeno.git
cd zeno
cmake -B build -DCMAKE_TOOLCHAIN_FILE=$PWD/../vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --parallel 8
