#include <zeno/zty/mesh/DCEL.h>
#include <zeno/zty/mesh/Mesh.h>
#include <unordered_map>
#include <stdexcept>
#include <vector>


ZENO_NAMESPACE_BEGIN
namespace zty {


DCEL::operator Mesh() const
{
    Mesh mesh;

    for (uint32_t v = 0; v < vert.size(); v++) {
        mesh.vert.push_back(vert[v].co);
    }

    for (uint32_t f = 0; f < face.size(); f++) {
        auto e = face[f].first;
        uint32_t npoly = 0;
        do {
            npoly++;
            auto l = edge[e].origin;
            mesh.loop.push_back(l);
            e = edge[e].next;
        } while (e != face[f].first);
        mesh.poly.push_back(npoly);
    }

    return mesh;
}


static inline uint32_t add(auto &a) {
    auto ret = static_cast<uint32_t>(a.size());
    a.emplace_back();
    return ret;
}


DCEL::DCEL(Mesh const &mesh)
{
    for (auto const &pos: mesh.vert) {
        auto v = add(vert);
        vert[v].co = pos;
    }

    struct MyHash {
        inline size_t operator()(std::pair<uint32_t, uint32_t> const &p) const {
            return std::hash<uint32_t>{}(p.first) ^ std::hash<uint32_t>{}(p.second);
        }
    };

    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, MyHash> edge_lut;

    size_t l0 = 0;
    for (auto const &nl: mesh.poly) {
        if (nl < 3) continue;

        uint32_t f = add(face);
        uint32_t e0 = add(edge), e1 = e0;
        auto v0 = mesh.loop[l0 + nl - 1];
        face[f].first = e0;
        edge[e0].face = f;
        edge[e0].origin = v0;

        [[unlikely]] if (!edge_lut.emplace(
            std::make_pair(
                mesh.loop[l0 + nl - 1],
                mesh.loop[l0]),
            e0).second)
            throw std::runtime_error("overlap edge");

        for (size_t l = l0 + 1; l < l0 + nl; l++) {
            auto e = add(edge);
            auto v = mesh.loop[l - 1];
            edge[e].face = f;
            edge[e].origin = v;

            [[unlikely]] if (!edge_lut.emplace(
                std::make_pair(
                    mesh.loop[l - 1],
                    mesh.loop[l]),
                e).second)
                throw std::runtime_error("overlap edge");

            edge[e1].next = e;
            e1 = e;
        }
        edge[e1].next = e0;

        l0 += nl;
    }

    for (auto const &[k, e]: edge_lut) {
        std::pair ik(k.second, k.first);
        auto it = edge_lut.find(ik);
        if (it != edge_lut.end()) {
            edge[e].twin = it->second;
        } else {
            auto e1 = add(edge);
            edge[e1].origin = k.second;
            edge[e1].next = kInvalid;
            edge[e1].face = kInvalid;
            edge[e1].twin = e;
            edge[e].twin = e1;
        }
    }
}


}
ZENO_NAMESPACE_END
