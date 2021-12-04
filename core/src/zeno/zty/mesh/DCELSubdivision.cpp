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
        if (edge_lut.contains(&e))
            continue;

        auto e0 = e.origin->co;
        auto e1 = e.twin->origin->co;

        math::vec3f epos;
        if (e.face && e.twin->face) {
            auto f0 = face_lut.at(e.face)->co;
            auto f1 = face_lut.at(e.twin->face)->co;
            epos = (f0 + f1 + e0 + e1) * 0.25f;
        } else {
            epos = (e0 + e1) * 0.5f;
        }

        vert_leaving.emplace(e.origin, &e);
        vert_leaving.emplace(e.twin->origin, e.twin);

        auto ve = &that.vert.emplace_back();
        ve->co = epos;
        edge_lut.emplace(&e, ve);
        edge_lut.emplace(e.twin, ve);
    }

    for (auto const &v: vert) {
        auto [it0, it1] = vert_leaving.equal_range(&v);

        uint32_t n = 0, nface = 0;
        math::vec3f vpos(0);
        math::vec3f favg(0);
        for (auto it = it0; it != it1; ++it) {
            auto e = it->second;
            vpos += e->origin->co + e->twin->origin->co;
            n++;
            if (e->face) {
                favg += face_lut.at(e->face)->co;
                nface++;
            }
        }

        [[unlikely]] if (!n || !nface)
            continue;

        vpos *= 1.f / n;
        vpos += favg * (1.f / nface);
        vpos += (n - 3) * v.co;
        vpos *= 1.f / n;

        auto vv = &that.vert.emplace_back();
        vv->co = vpos;
        vert_lut.emplace(&v, vv);
    }

    for (auto const &[f, vf]: face_lut) {
        auto e0 = f->first, e = e0;

        auto lve = edge_lut.at(e);
        auto lte0 = &that.edge.emplace_back();
        auto lte1 = &that.edge.emplace_back();
        auto lte3 = &that.edge.emplace_back();
        lte0->twin = lte1;
        lte1->twin = lte0;
        lte0->origin = const_cast<Vert *>(vf);
        lte1->origin = const_cast<Vert *>(lve);
        lte3->origin = const_cast<Vert *>(lve);

        auto ltf = &that.face.emplace_back();
        ltf->first = lte0;
        lte0->face = ltf;
        // lte1->face = lltf;
        lte0->next = lte3;
        // lte1->next = llte0;
        lte3->face = ltf;

        auto lte2 = &that.edge.emplace_back();
        // lte2->origin = const_cast<Vert *>(llve);
        // lte2->face = lltf;
        lte2->next = lte1;
        // llte3->next = lte2;

        auto olte1 = lte1;
        auto olte2 = lte2;

        e = e->next;

        while (e != e0) {
            auto ve = edge_lut.at(e);
            auto te0 = &that.edge.emplace_back();
            auto te1 = &that.edge.emplace_back();
            auto te3 = &that.edge.emplace_back();
            te0->twin = te1;
            te1->twin = te0;
            te0->origin = const_cast<Vert *>(vf);
            te1->origin = const_cast<Vert *>(ve);
            te3->origin = const_cast<Vert *>(ve);

            auto tf = &that.face.emplace_back();
            tf->first = te0;
            te0->face = tf;
            te1->face = ltf;
            te0->next = te3;
            te1->next = lte0;
            te3->face = tf;

            auto te2 = &that.edge.emplace_back();
            te2->origin = const_cast<Vert *>(lve);
            te2->face = ltf;
            te2->next = te1;
            lte3->next = te2;

            lve = ve;
            lte0 = te0;
            lte1 = te1;
            lte3 = te3;
            ltf = tf;
            lte2 = te2;

            e = e->next;
        }

        olte1->face = ltf;
        olte1->next = lte0;
        olte2->origin = const_cast<Vert *>(lve);
        olte2->face = ltf;
        lte3->next = olte2;
    }

    return that;
}


}
ZENO_NAMESPACE_END
