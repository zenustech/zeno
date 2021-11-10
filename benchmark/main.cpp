#include <zeno/dop/dop.h>
#include <zeno/types/Mesh.h>
#include <zeno/types/MeshIO.h>
#include <zeno/types/MeshTriangulate.h>
#include <benchmark/benchmark.h>
#include <fstream>

USING_ZENO_NAMESPACE

static types::Mesh getTestMesh() {
    types::Mesh mesh;
    std::ifstream fin("models/cube.obj");
    types::readMeshFromOBJ(fin, mesh);
    return mesh;
}

static void BM_meshToTriangles(benchmark::State &state) {
    types::Mesh mesh = getTestMesh();

    for (auto _: state) {
        auto tris = types::meshToTriangles(mesh);
        zycl::host_get(tris);
    }
}
BENCHMARK(BM_meshToTriangles);
