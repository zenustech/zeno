#include <zeno/zty/mesh/MeshTriangulate.h>


ZENO_NAMESPACE_BEGIN
namespace zty {


void meshTriangulate(Mesh &mesh) {

    std::vector<math::vec2i> poly;
    std::vector<int> loop;

    poly.reserve(mesh.poly.size() * 3);
    loop.reserve(mesh.poly.size() * 3);

    size_t index = 0;
    std::for_each(begin(mesh.poly), end(mesh.poly), [&] (auto const &p) {
        auto const &[p_start, p_num] = p;
        if (p_num <= 2) return;
        int first = mesh.loop[p_start];
        int last = mesh.loop[p_start + 1];
        for (int l = p_start + 2; l < p_start + p_num; l++) {
            int now = mesh.loop[l];
            last = now;
            poly.emplace_back(index * 3, 3);
            loop.push_back(first);
            loop.push_back(last);
            loop.push_back(now);
        }
        index++;
    });

    mesh.poly = std::move(poly);
    mesh.loop = std::move(loop);
}


std::vector<math::vec3f> meshToTriangles(Mesh const &mesh) {
    std::vector<math::vec3i> indices;
    indices.reserve(mesh.poly.size() * 3);

    std::for_each(begin(mesh.poly), end(mesh.poly), [&] (auto const &poly) {
        auto const &[p_start, p_num] = poly;
        if (p_num <= 2) return;
        int first = mesh.loop[p_start];
        int last = mesh.loop[p_start + 1];
        for (int l = p_start + 2; l < p_start + p_num; l++) {
            int now = mesh.loop[l];
            indices.emplace_back(first, last, now);
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
