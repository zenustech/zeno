// https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface
#include <zeno/zty/mesh/DCEL.h>
#include <zeno/zty/mesh/Mesh.h>
#include <unordered_map>
#include <optional>
#include <vector>


ZENO_NAMESPACE_BEGIN
namespace zty {


DCEL DCEL::subdivision()
{
    DCEL that;
    std::unordered_map<Vert const *, Vert *> vert_lut;
    std::unordered_map<Edge const *, Vert *> edge_lut;
    std::unordered_map<Face const *, Vert *> face_lut;

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
        auto eqr = vert_leaving.equal_range(&v);
        [[unlikely]] if (eqr.first == eqr.second)
            continue;

        auto vpos = [&] {
            uint32_t n = 0;
            math::vec3f vpos(0);
            math::vec3f favg(0);
            for (auto it = eqr.first; it != eqr.second; ++it) {
                auto e = it->second;
                vpos += e->origin->co + e->twin->origin->co;
                [[likely]] if (e->face) {
                    favg += face_lut.at(e->face)->co;
                } else {
                    uint32_t n = 0, ne = 0;
                    math::vec3f vpos(0);
                    for (auto it = eqr.first; it != eqr.second; ++it) {
                        auto e = it->second;
                        if (!e->face || !e->twin->face) {
                            vpos += e->origin->co + e->twin->origin->co;
                            ne++;
                        }
                        n++;
                    }
                    [[likely]] if (ne)
                        vpos *= 3.f / ne;
                    vpos += (n - 1.5f) * v.co;
                    vpos *= 1.f / n;
                    return vpos;
                }
                n++;
            }

            vpos += favg;
            vpos *= 1.f / n;
            vpos += (n - 3) * v.co;
            vpos *= 1.f / n;
            return vpos;
        }();

        auto vv = &that.vert.emplace_back();
        vv->co = vpos;
        vert_lut.emplace(&v, vv);
    }

    std::unordered_map<Edge const *, Edge *> tte2_lut;
    std::unordered_map<Edge const *, Edge *> te3_lut;
    for (auto const &[f, vf]: face_lut) {
        auto e0 = f->first, e = e0;

        auto lve = edge_lut.at(e);
        auto lte0 = &that.edge.emplace_back();
        auto lte1 = &that.edge.emplace_back();
        auto lte3 = &that.edge.emplace_back();
        te3_lut.emplace(e, lte3);
        lte0->twin = lte1;
        lte1->twin = lte0;
        lte0->origin = vf;
        lte1->origin = lve;
        lte3->origin = lve;

        auto ltf = &that.face.emplace_back();
        ltf->first = lte0;
        lte0->face = ltf;
        lte0->next = lte3;
        lte3->face = ltf;

        auto lte2 = &that.edge.emplace_back();
        tte2_lut.emplace(e->twin, lte2);
        lte2->origin = vert_lut.at(e->origin);
        lte2->next = lte1;

        auto olte1 = lte1;
        auto olte2 = lte2;

        e = e->next;

        while (e != e0) {
            auto ve = edge_lut.at(e);
            auto te0 = &that.edge.emplace_back();
            auto te1 = &that.edge.emplace_back();
            auto te3 = &that.edge.emplace_back();
            te3_lut.emplace(e, te3);
            te0->twin = te1;
            te1->twin = te0;
            te0->origin = vf;
            te1->origin = ve;
            te3->origin = ve;

            auto tf = &that.face.emplace_back();
            tf->first = te0;
            te0->face = tf;
            te1->face = ltf;
            te0->next = te3;
            te1->next = lte0;
            te3->face = tf;

            auto te2 = &that.edge.emplace_back();
            tte2_lut.emplace(e->twin, te2);
            te2->origin = vert_lut.at(e->origin);
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
        olte2->face = ltf;
        lte3->next = olte2;
    }

    for (auto const &[e, te3]: te3_lut) {
        if (auto it = tte2_lut.find(e); it != tte2_lut.end()) {
            auto te2 = it->second;
            te2->twin = te3;
            te3->twin = te2;
            tte2_lut.erase(it);
        } else {
            auto te2 = &that.edge.emplace_back();
            te2->next = nullptr;
            te2->face = nullptr;
            te2->twin = te3;
            te3->twin = te2;
        }
    }

    for (auto const &[et, te2]: tte2_lut) {
        auto te3 = &that.edge.emplace_back();
        te3->next = nullptr;
        te3->face = nullptr;
        te3->twin = te2;
        te2->twin = te3;
    }

    return that;
}


}
ZENO_NAMESPACE_END
