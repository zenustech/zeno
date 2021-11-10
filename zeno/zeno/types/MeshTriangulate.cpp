#include <zeno/types/MeshTriangulate.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for_each.h>


ZENO_NAMESPACE_BEGIN
namespace types {


std::vector<math::vec3f> meshToTriangles(Mesh const &mesh) {

#if 0
    tbb::concurrent_vector<math::vec3i> indices;
#else
    std::vector<math::vec3i> indices;
#endif
    indices.reserve(mesh.poly.size() * 3);

#if 0
    tbb::parallel_for_each(begin(mesh.poly), end(mesh.poly), [&] (auto const &poly) {
#else
    std::for_each(begin(mesh.poly), end(mesh.poly), [&] (auto const &poly) {
#endif
        auto const &[p_start, p_num] = poly;
        if (p_num <= 2) return;
        int first = mesh.loop[p_start];
        int last = mesh.loop[p_start + 1];
        for (int l = p_start + 2; l < p_start + p_num; l++) {
            int now = mesh.loop[l];
            indices.push_back({first, last, now});
            last = now;
        }
    });

    std::vector<math::vec3f> vertices;
    for (auto const &[i, j, k]: indices) {
        vertices.push_back(mesh.vert[i]);
        vertices.push_back(mesh.vert[j]);
        vertices.push_back(mesh.vert[k]);
    }

    return vertices;
}


}
ZENO_NAMESPACE_END
