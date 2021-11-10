#include <zeno/dop/dop.h>
#include <zeno/types/Mesh.h>
#include <zeno/types/MeshIO.h>
#include <zeno/types/MeshTriangulate.h>
#include <benchmark/benchmark.h>
#include <fstream>

USING_ZENO_NAMESPACE

static types::Mesh getTestMesh() {
    types::Mesh mesh;
    std::ifstream fin("models/Pig_Head.obj");
    types::readMeshFromOBJ(fin, mesh);
    return mesh;
}

static void BM_meshToTriangles(benchmark::State &state) {
    types::Mesh mesh = getTestMesh();

    zycl::vector<math::vec3f> tris;
    for (auto _: state) {
        tris = types::meshToTriangles(mesh);
        benchmark::DoNotOptimize(tris);
    }
    zycl::host_get(tris);
}
BENCHMARK(BM_meshToTriangles)
    ->ComputeStatistics("max", [] (std::vector<double> const &v) -> double {
        return *(std::max_element(std::begin(v), std::end(v)));
    })
    ->ComputeStatistics("min", [] (std::vector<double> const &v) -> double {
        return *(std::min_element(std::begin(v), std::end(v)));
    })
;
