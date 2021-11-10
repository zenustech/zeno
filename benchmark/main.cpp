#include <zeno/dop/dop.h>
#include <zeno/types/Mesh.h>
#include <zeno/types/MeshIO.h>
#include <zeno/types/MeshTriangulate.h>
#include <benchmark/cppbenchmark.h>
#include <fstream>

USING_ZENO_NAMESPACE

BENCHMARK("meshToTriangles")
{
    types::Mesh mesh;
    {
        std::ifstream fin("models/cube.obj");
        types::readMeshFromOBJ(fin, mesh);
    }

    auto tris = types::meshToTriangles(mesh);
    zycl::host_get(tris);
}

BENCHMARK_MAIN();
