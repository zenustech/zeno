cmake -B build -DCMAKE_BUILD_TYPE=Debug -DZENO_WITH_zenvdb:BOOL=ON -DZENO_WITH_ZenoFX:BOOL=ON -DZENO_ENABLE_OPTIX:BOOL=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1

cmake --build build --parallel $(nproc)

ln -s ./build/compile_commands.json ./
