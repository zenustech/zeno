// https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface
#include <zeno/zty/mesh/DCEL.h>
#include <zeno/zty/mesh/Mesh.h>
#include <unordered_map>
#include <vector>
#include <tuple>


ZENO_NAMESPACE_BEGIN
namespace zty {


DCEL DCEL::subdivision()
{
    DCEL that;
    std::unordered_map<Vert const *, Vert const *> vert_lut;
    std::unordered_map<Edge const *, Vert const *> edge_lut;
    std::unordered_map<Face const *, Vert const *> face_lut;

    for (auto const &f: face) {
        auto e = f.first;
        auto favg = e->origin->co;
        size_t fnum = 1;
        e = e->next;
        while (e != f.first) {
            favg += e->origin->co;
            e = e->next;
            fnum++;
        }
        favg *= 1.f / fnum;

        auto vf = &that.vert.emplace_back();
        vf->co = favg;
        face_lut.emplace(&f, vf);
    }

    std::unordered_multimap<Vert const *, Edge const *> vert_leaving;
    for (auto const &e: edge) {
        auto f0 = face_lut.at(e.face)->co;
        auto f1 = face_lut.at(e.twin->face)->co;
        auto e0 = e.origin->co;
        auto e1 = e.twin->origin->co;
        auto epos = (f0 + f1 + e0 + e1) * 0.25f;
        vert_leaving.emplace(e.origin, &e);

        if (!edge_lut.contains(&e)) {
            auto ve = &that.vert.emplace_back();
            ve->co = epos;
            edge_lut.emplace(&e, ve);
            edge_lut.emplace(e.twin, ve);
        }
    }

    for (auto const &v: vert) {
        auto [it0, it1] = vert_leaving.equal_range(&v);

        uint32_t n = 0;
        math::vec3f vpos(0);
        for (auto it = it0; it != it1; ++it) {
            auto e = it->second;
            auto eavg_x2 = e->origin->co + e->twin->origin->co;
            auto favg = face_lut.at(e->face)->co;
            vpos += eavg_x2 + favg;
            n++;
        }

        vpos *= 1.f / n;
        vpos += (n - 3) * v.co;
        vpos *= 1.f / n;

        auto vv = &that.vert.emplace_back();
        vv->co = vpos;
        vert_lut.emplace(&v, vv);
    }

    return that;
}


}
ZENO_NAMESPACE_END
