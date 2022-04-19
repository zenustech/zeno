#!/bin/bash
set -e

# NOTE: cgmesh must be ON when FEM or Skinning is ON, this is to build the fucking libigl which is now self-contained in cgmesh

cmake -G Ninja -B build -Wno-dev -DCMAKE_BUILD_TYPE=Release -DZENO_NO_WARNING:BOOL=ON -DZENO_BUILD_SHARED:BOOL=ON -DZENO_MULTIPROCESS:BOOL=ON -DZENO_INSTALL_TARGET:BOOL=ON -DZENO_WITH_ZenoFX:BOOL=ON -DZENO_WITH_zenvdb:BOOL=ON -DZENO_WITH_FastFLIP:BOOL=ON -DZENO_WITH_FEM:BOOL=ON -DZENO_WITH_Rigid:BOOL=ON -DZENO_WITH_cgmesh:BOOL=ON -DZENO_WITH_oldzenbase:BOOL=ON -DZENO_WITH_TreeSketch:BOOL=ON -DZENO_WITH_Skinning:BOOL=ON -DZENO_WITH_LSystem:BOOL=ON -DZENO_WITH_Alembic:BOOL=ON
cmake --build build --parallel 12
