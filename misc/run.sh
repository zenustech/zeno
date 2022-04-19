#!/bin/bash
set -e

#
# This is a fking script to make zhxx very happy.
#
# Run 'misc/run.sh' to build and run this smart project in one command to increase jiafangbaba happiness!
#
# The cgmesh must be ON when FEM or Skinning is ON, to get the fking libigl which is now contained in cgmesh.
#
# For yin ♂ wei ♂ da users, please run 'misc/run.sh ywd' for CUDA support!
#

cmake -G Ninja -B /tmp/zeno-build -Wno-dev -DCMAKE_BUILD_TYPE=Release -DZENO_WITH_ZenoFX:BOOL=ON -DZENOFX_ENABLE_OPENVDB:BOOL=ON -DZENOFX_ENABLE_LBVH:BOOL=ON -DZENO_WITH_zenvdb:BOOL=ON -DZENO_WITH_FastFLIP:BOOL=ON -DZENO_WITH_FEM:BOOL=ON -DZENO_WITH_Rigid:BOOL=ON -DZENO_WITH_cgmesh:BOOL=ON -DZENO_WITH_oldzenbase:BOOL=ON -DZENO_WITH_TreeSketch:BOOL=ON -DZENO_WITH_Skinning:BOOL=ON -DZENO_WITH_Euler:BOOL=ON -DZENO_WITH_Functional:BOOL=ON -DZENO_WITH_LSystem:BOOL=ON -DZENO_WITH_Alembic:BOOL=ON ${1+-DZENO_WITH_gmpm:BOOL=ON -DZENO_WITH_mesher:BOOL=ON}
cmake --build /tmp/zeno-build --parallel 12
/tmp/zeno-build/bin/zenoedit
