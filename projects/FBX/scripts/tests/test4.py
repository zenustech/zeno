import os
import sys
import shlex
import subprocess

cmd = "cmake -S/opt/src/zeno -B/opt/src/zeno/build -DCMAKE_TOOLCHAIN_FILE=/home/zenus/work/vcpkg/scripts/buildsystems/vcpkg.cmake -GNinja -Wno-dev -DZENOFX_ENABLE_OPENVDB:BOOL=ON -DZENOFX_ENABLE_LBVH:BOOL=ON -DZENO_ENABLE_OPTIX:BOOL=ON -DZENO_SYSTEM_OPENVDB:BOOL=OFF -DZENO_MULTIPROCESS=ON -DCMAKE_BUILD_TYPE=Release -DZENO_WITH_ZenoFX:BOOL=ON -DZENO_WITH_zenvdb:BOOL=ON -DZENO_WITH_FastFLIP:BOOL=ON -DZENO_WITH_Rigid:BOOL=ON -DZENO_WITH_oldzenbase:BOOL=ON -DZENO_WITH_TreeSketch:BOOL=ON -DZENO_WITH_Functional:BOOL=ON -DZENO_WITH_Alembic:BOOL=ON -DZENO_WITH_FBX:BOOL=ON -DZENO_WITH_CalcGeometryUV:BOOL=ON -DZENO_WITH_MeshSubdiv:BOOL=ON -DZENO_WITH_CuLagrange:BOOL=ON -DZENO_WITH_CuEulerian:BOOL=ON -DZENO_WITH_CUDA:BOOL=ON -DZENO_WITH_TOOL_FLIPtools:BOOL=ON -DZENO_WITH_TOOL_BulletTools:BOOL=ON -DZENO_WITH_Python:BOOL=OFF -DZENO_ENABLE_OPENMP:BOOL=ON"

print("Shlex", shlex.split(cmd))

# # r = os.system(cmd)
# r = os.popen(cmd).readlines()
# for l in r:
#     print("-------", l)


process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)

while True:
    line = process.stdout.readline().decode("utf-8")
    if line:
        sys.stdout.write(line)

    if process.poll() is not None:
        print("Run Process Poll")
        break