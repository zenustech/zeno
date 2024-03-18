[ "$1" ] && [ $1 = full ] && pack="full" || pack="mini"

[ "$2" ] && [ $2 = debug ] && flag="debug" || flag="release"
echo $flag

if [ $pack != "full" ]; then

echo "making minimum build..."

cmake -B build -DCMAKE_BUILD_TYPE=$flag \
    -DZENO_MULTIPROCESS=ON \
    -DZENO_WITH_zenvdb:BOOL=ON \
    -DZENO_SYSTEM_OPENVDB=OFF \
    -DZENO_WITH_ZenoFX:BOOL=ON \
    -DZENO_ENABLE_OPTIX:BOOL=ON \
    -DZENO_WITH_FBX:BOOL=ON \
    -DZENO_WITH_Alembic:BOOL=ON \
    -DZENO_WITH_MeshSubdiv:BOOL=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \

else

echo "making full build..."

cmake -B build -DCMAKE_BUILD_TYPE=$flag \
    -DZENO_WITH_ZenoFX:BOOL=ON \
    -DZENO_ENABLE_OPTIX:BOOL=ON \
    -DZENO_SYSTEM_OPENVDB=OFF \
    -DZENO_WITH_zenvdb:BOOL=ON \
    -DZENOFX_ENABLE_OPENVDB:BOOL=ON \
    -DZENOFX_ENABLE_LBVH:BOOL=ON \
    -DZENO_WITH_FastFLIP:BOOL=ON \
    -DZENO_WITH_FEM:BOOL=ON \
    -DZENO_WITH_Rigid:BOOL=ON \
    -DZENO_WITH_cgmesh:BOOL=ON \
    -DZENO_WITH_oldzenbase:BOOL=ON \
    -DZENO_WITH_TreeSketch:BOOL=ON \
    -DZENO_WITH_Skinning:BOOL=ON \
    -DZENO_WITH_Euler:BOOL=ON \
    -DZENO_WITH_Functional:BOOL=ON \
    -DZENO_WITH_LSystem:BOOL=ON \
    -DZENO_WITH_mesher:BOOL=ON \
    -DZENO_WITH_Alembic:BOOL=ON \
    -DZENO_WITH_FBX:BOOL=ON \
    -DZENO_WITH_DemBones:BOOL=ON \
    -DZENO_WITH_SampleModel:BOOL=ON \
    -DZENO_WITH_CalcGeometryUV:BOOL=ON \
    -DZENO_WITH_MeshSubdiv:BOOL=ON \
    -DZENO_WITH_Audio:BOOL=ON \
    -DZENO_WITH_PBD:BOOL=ON \
    -DZENO_WITH_GUI:BOOL=ON \
    -DZENO_WITH_ImgCV:BOOL=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
fi

cmake --build build --parallel $(nproc) \

# ln -s ./build/compile_commands.json ./
cp ./build/compile_commands.json ./
rm -rf /var/tmp/OptixCache_$USER/