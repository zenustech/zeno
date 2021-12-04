// https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface
#include <zeno/zty/mesh/DCEL.h>
#include <zeno/zty/mesh/Mesh.h>
#include <unordered_map>
#include <optional>
#include <vector>


ZENO_NAMESPACE_BEGIN
namespace zty {


static inline uint32_t add(auto &a) {
    auto ret = static_cast<uint32_t>(a.size());
    a.emplace_back();
    return ret;
}


DCEL DCEL::subdivision()
{
    DCEL that;
    std::unordered_map<uint32_t, uint32_t> vert_lut;
    std::unordered_map<uint32_t, uint32_t> edge_lut;
    std::unordered_map<uint32_t, uint32_t> face_lut;

    for (uint32_t f = 0; f < face.size(); f++) {
        auto e = face[f].first;
        auto favg = vert[edge[e].origin].co;
        size_t fnum = 1;
        e = edge[e].next;
        while (e != face[f].first) {
            favg += vert[edge[e].origin].co;
            e = edge[e].next;
            fnum++;
        }
        favg *= 1.f / fnum;

        auto vf = add(that.vert);
        vert[vf].co = favg;
        face_lut.emplace(f, vf);
    }

    std::unordered_multimap<uint32_t, uint32_t> vert_leaving;
    for (uint32_t e = 0; e < edge.size(); e++) {
        if (edge_lut.contains(e))
            continue;

        auto e0 = vert[edge[e].origin].co;
        auto e1 = vert[edge[edge[e].twin].origin].co;

        math::vec3f epos;
        if (edge[e].face && edge[edge[e].twin].face) {
            auto f0 = vert[face_lut.at(edge[e].face)].co;
            auto f1 = vert[face_lut.at(edge[edge[e].twin].face)].co;
            epos = (f0 + f1 + e0 + e1) * 0.25f;
        } else {
            epos = (e0 + e1) * 0.5f;
        }

        vert_leaving.emplace(edge[e].origin, e);
        vert_leaving.emplace(edge[edge[e].twin].origin, edge[e].twin);

        auto ve = add(that.vert);
        vert[ve].co = epos;
        edge_lut.emplace(e, ve);
        edge_lut.emplace(edge[e].twin, ve);
    }

    for (uint32_t v = 0; v < vert.size(); v++) {
        auto eqr = vert_leaving.equal_range(v);
        [[unlikely]] if (eqr.first == eqr.second)
            continue;

        auto vpos = [&] {
            uint32_t n = 0;
            math::vec3f vpos(0);
            math::vec3f favg(0);
            for (auto it = eqr.first; it != eqr.second; ++it) {
                auto e = it->second;
                vpos += vert[edge[e].origin].co + vert[edge[edge[e].twin].origin].co;
                [[likely]] if (edge[e].face != kInvalid) {
                    favg += vert[face_lut.at(edge[e].face)].co;
                } else {
                    uint32_t n = 0, ne = 0;
                    math::vec3f vpos(0);
                    for (auto it = eqr.first; it != eqr.second; ++it) {
                        auto e = it->second;
                        if (edge[e].face == kInvalid || edge[edge[e].twin].face == kInvalid) {
                            vpos += vert[edge[e].origin].co + vert[edge[edge[e].twin].origin].co;
                            ne++;
                        }
                        n++;
                    }
                    [[likely]] if (ne)
                        vpos *= 3.f / ne;
                    vpos += (n - 1.5f) * vert[v].co;
                    vpos *= 1.f / n;
                    return vpos;
                }
                n++;
            }

            vpos += favg;
            vpos *= 1.f / n;
            vpos += (n - 3) * vert[v].co;
            vpos *= 1.f / n;
            return vpos;
        }();

        auto vv = add(that.vert);
        vert[vv].co = vpos;
        vert_lut.emplace(v, vv);
    }

    std::unordered_map<uint32_t, uint32_t> tte2_lut;
    std::unordered_map<uint32_t, uint32_t> te3_lut;
    for (auto const &[f, vf]: face_lut) {
        auto e0 = face[f].first, e = e0;

        auto lve = edge_lut.at(e);
        auto lte0 = add(that.edge);
        auto lte1 = add(that.edge);
        auto lte3 = add(that.edge);
        te3_lut.emplace(e, lte3);
        edge[lte0].twin = lte1;
        edge[lte1].twin = lte0;
        edge[lte0].origin = vf;
        edge[lte1].origin = lve;
        edge[lte3].origin = lve;

        auto ltf = add(that.face);
        face[ltf].first = lte0;
        edge[lte0].face = ltf;
        edge[lte0].next = lte3;
        edge[lte3].face = ltf;

        auto lte2 = add(that.edge);
        tte2_lut.emplace(edge[e].twin, lte2);
        edge[lte2].origin = vert_lut.at(edge[e].origin);
        edge[lte2].next = lte1;

        auto olte1 = lte1;
        auto olte2 = lte2;

        e = edge[e].next;

        while (e != e0) {
            auto ve = edge_lut.at(e);
            auto te0 = add(that.edge);
            auto te1 = add(that.edge);
            auto te3 = add(that.edge);
            te3_lut.emplace(e, te3);
            edge[te0].twin = te1;
            edge[te1].twin = te0;
            edge[te0].origin = vf;
            edge[te1].origin = ve;
            edge[te3].origin = ve;

            auto tf = add(that.face);
            face[tf].first = te0;
            edge[te0].face = tf;
            edge[te1].face = ltf;
            edge[te0].next = te3;
            edge[te1].next = lte0;
            edge[te3].face = tf;

            auto te2 = add(that.edge);
            tte2_lut.emplace(edge[e].twin, te2);
            edge[te2].origin = vert_lut.at(edge[e].origin);
            edge[te2].face = ltf;
            edge[te2].next = te1;
            edge[lte3].next = te2;

            lve = ve;
            lte0 = te0;
            lte1 = te1;
            lte3 = te3;
            ltf = tf;
            lte2 = te2;

            e = edge[e].next;
        }

        edge[olte1].face = ltf;
        edge[olte1].next = lte0;
        edge[olte2].face = ltf;
        edge[lte3].next = olte2;
    }

    for (auto const &[e, te3]: te3_lut) {
        if (auto it = tte2_lut.find(e); it != tte2_lut.end()) {
            auto te2 = it->second;
            edge[te2].twin = te3;
            edge[te3].twin = te2;
            tte2_lut.erase(it);
        } else {
            auto te2 = add(that.edge);
            edge[te2].next = kInvalid;
            edge[te2].face = kInvalid;
            edge[te2].twin = te3;
            edge[te3].twin = te2;
        }
    }

    for (auto const &[et, te2]: tte2_lut) {
        auto te3 = add(that.edge);
        edge[te3].next = kInvalid;
        edge[te3].face = kInvalid;
        edge[te3].twin = te2;
        edge[te2].twin = te3;
    }

    return that;
}


}
ZENO_NAMESPACE_END
