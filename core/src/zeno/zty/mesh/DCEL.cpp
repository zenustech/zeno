#include <zeno/zty/mesh/DCEL.h>
#include <zeno/zty/mesh/Mesh.h>
#include <unordered_map>


ZENO_NAMESPACE_BEGIN
namespace zty {


DCEL::DCEL() noexcept = default;
DCEL::DCEL(DCEL &&that) noexcept = default;
DCEL &DCEL::operator=(DCEL &&that) noexcept = default;


DCEL &DCEL::operator=(DCEL const &that) {
    return this->operator=(DCEL(that));
}


DCEL::DCEL(DCEL const &that)
{
    vert.clear();
    edge.clear();
    face.clear();

    std::unordered_map<Edge const *, Edge *> edge_lut;
    std::unordered_map<Vert const *, Vert *> vert_lut;
    std::unordered_map<Face const *, Face *> face_lut;

    for (auto const &o: that.vert) {
        auto n = &vert.emplace_back(o);
        vert_lut.emplace(&o, n);
    }

    for (auto const &o: that.edge) {
        auto n = &edge.emplace_back(o);
        edge_lut.emplace(&o, n);
    }

    for (auto const &o: that.face) {
        auto n = &face.emplace_back(o);
        face_lut.emplace(&o, n);
    }

    for (auto &o: edge) {
        o.origin = vert_lut.at(o.origin);
        o.twin = edge_lut.at(o.twin);
        o.next = edge_lut.at(o.next);
        o.face = face_lut.at(o.face);
    }

    for (auto &o: face) {
        o.first = edge_lut.at(o.first);
    }
}


}
ZENO_NAMESPACE_END
