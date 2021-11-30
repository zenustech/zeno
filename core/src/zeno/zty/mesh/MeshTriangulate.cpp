#include <zeno/zty/mesh/MeshTriangulate.h>


ZENO_NAMESPACE_BEGIN
namespace zty {


void meshTriangulate(Mesh &mesh) {
    std::vector<uint32_t> poly;
    std::vector<uint32_t> loop;

    poly.reserve(mesh.poly.size() * 3);
    loop.reserve(mesh.poly.size() * 3);

    size_t start = 0;
    std::for_each(begin(mesh.poly), end(mesh.poly), [&] (uint32_t npoly) {
        if (npoly >= 3) {
            int first = mesh.loop[start];
            int last = mesh.loop[start + 1];
            for (size_t l = start + 2; l < start + npoly; l++) {
                int now = mesh.loop[l];
                poly.push_back(3);
                loop.push_back(first);
                loop.push_back(last);
                loop.push_back(now);
                last = now;
            }
        }
        start += npoly;
    });

    mesh.poly = std::move(poly);
    mesh.loop = std::move(loop);
}


std::vector<math::vec3f> meshToTriangles(Mesh const &mesh) {
    std::vector<math::vec3f> vertices;
    vertices.reserve(mesh.poly.size() * 3);

    size_t start = 0;
    std::for_each(begin(mesh.poly), end(mesh.poly), [&] (size_t npoly) {
        if (npoly >= 3) {
            int first = mesh.loop[start];
            int last = mesh.loop[start + 1];
            for (int l = start + 2; l < start + npoly; l++) {
                int now = mesh.loop[l];
                vertices.push_back(mesh.vert[first]);
                vertices.push_back(mesh.vert[last]);
                vertices.push_back(mesh.vert[now]);
                last = now;
            }
        }
        start += npoly;
    });

    return vertices;
}


}
ZENO_NAMESPACE_END
