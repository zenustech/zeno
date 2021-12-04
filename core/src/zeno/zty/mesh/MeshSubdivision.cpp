#include <zeno/zty/mesh/MeshSubdivision.h>
#include <unordered_map>
#include <stdexcept>


ZENO_NAMESPACE_BEGIN
namespace zty {


void meshSubdivisionSimple(Mesh &mesh)
{
#if 0
    std::vector<uint32_t> vert_leaving(mesh.vert.size());

    std::vector<math::vec3f> face_avg;
    face_avg.reserve(mesh.poly.size());

    size_t l0 = 0;
    for (auto const &nl: mesh.poly) {
        [[unlikely]] if (nl < 3)
            throw std::runtime_error("polygon with less than 3 edges");

        auto avg = mesh.vert[mesh.loop[l0]];
        for (uint32_t l = 1; l < nl; l++) {
            avg += mesh.vert[mesh.loop[l0 + l]];
        }
        avg *= 1.f / nl;
        face_avg.push_back(avg);

        l0 += nl;
    }

    std::unordered_map<Edge const *, math::vec3f> edge_point;
    for (auto const &e: edge) {
        auto f0 = face_avg.at(e.face);
        auto f1 = face_avg.at(e.twin->face);
        auto e0 = e.origin->co;
        auto e1 = e.twin->origin->co;
        auto ep = (f0 + f1 + e0 + e1) * 0.25f;
        edge_point.emplace(&e, ep);
        vert_leaving.emplace(e.origin, &e);
    }

    for (auto &v: vert) {
        auto [it0, it1] = vert_leaving.equal_range(&v);
        uint32_t n = 0;
        math::vec3f res(0);
        for (auto it = it0; it != it1; ++it) {
            auto e = it->second;
            auto ep2 = e->origin->co + e->twin->origin->co;
            auto fp = face_avg.at(e->face);
            res += ep2 + fp;
            n++;
        }
        res += (n - 3) * v.co;
        res *= 1.f / n;
        v.co = res;
    }

    // TODO
#endif
}


}
ZENO_NAMESPACE_END
