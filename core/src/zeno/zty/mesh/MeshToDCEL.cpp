#include <zeno/zty/mesh/DCEL.h>
#include <zeno/zty/mesh/Mesh.h>
#include <unordered_map>
#include <stdexcept>
#include <vector>


ZENO_NAMESPACE_BEGIN
namespace zty {


DCEL::DCEL(Mesh const &mesh)
{
    std::vector<Vert *> vert_lut;
    vert_lut.reserve(mesh.vert.size());
    for (auto const &pos: mesh.vert) {
        auto v = &vert.emplace_back();
        v->co[0] = pos[0];
        v->co[1] = pos[1];
        v->co[2] = pos[2];
        vert_lut.push_back(v);
    }

    struct MyHash {
        size_t operator()(std::pair<uint32_t, uint32_t> const &p) const {
            return std::hash<uint32_t>{}(p.first) ^ std::hash<uint32_t>{}(p.second);
        }
    };

    std::unordered_map<std::pair<uint32_t, uint32_t>, Edge *, MyHash> edge_lut;

    size_t l0 = 0;
    for (auto const &nl: mesh.poly) {
        if (nl < 3) continue;

        auto f = &face.emplace_back();
        auto e0 = &edge.emplace_back(), e1 = e0;
        auto v0 = vert_lut.at(mesh.loop[l0 + nl - 1]);
        f->first = e0;
        e0->face = f;
        e0->origin = v0;
        bool succ = edge_lut.emplace(
            std::make_pair(
                mesh.loop[l0 + nl - 1],
                mesh.loop[l0]),
            e0).second;
        [[unlikely]] if (!succ)
            throw std::runtime_error("overlap edge");

        for (size_t l = l0 + 1; l < l0 + nl; l++) {
            auto e = &edge.emplace_back();
            auto v = vert_lut.at(mesh.loop[l - 1]);
            e->face = f;
            e->origin = v;
            bool succ = edge_lut.emplace(
                std::make_pair(
                    mesh.loop[l - 1],
                    mesh.loop[l]),
                e).second;
            [[unlikely]] if (!succ)
                throw std::runtime_error("overlap edge");
            e1->next = e;
            e1 = e;
        }
        e1->next = e0;

        l0 += nl;
    }

    for (auto const &[k, e]: edge_lut) {
        std::pair ik(k.second, k.first);
        auto it = edge_lut.find(ik);
        if (it != edge_lut.end()) {
            e->twin = it->second;
        } else {
            auto e1 = &edge.emplace_back();
            e1->origin = vert_lut.at(k.second);
            e1->next = nullptr;  // should inf-face have loop ptr?
            e1->face = nullptr;
            e1->twin = e;
            e->twin = e1;
        }
    }
}


}
ZENO_NAMESPACE_END
