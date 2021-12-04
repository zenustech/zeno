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

    std::vector<uint32_t> face_lut(face.size());

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
        that.vert[vf].co = favg;
        face_lut[f] = vf;
    }

    std::vector<uint32_t> edge_lut(edge.size(), kInvalid);
    std::unordered_multimap<uint32_t, uint32_t> vert_leaving;
    for (uint32_t e = 0; e < edge.size(); e++) {
        if (edge_lut[e] != kInvalid)
            continue;

        auto e0 = vert[edge[e].origin].co;
        auto e1 = vert[edge[edge[e].twin].origin].co;

        math::vec3f epos;
        if (edge[e].face != kInvalid && edge[edge[e].twin].face != kInvalid) {
            auto f0 = that.vert[face_lut.at(edge[e].face)].co;
            auto f1 = that.vert[face_lut.at(edge[edge[e].twin].face)].co;
            epos = (f0 + f1 + e0 + e1) * 0.25f;
        } else {
            epos = (e0 + e1) * 0.5f;
        }

        vert_leaving.emplace(edge[e].origin, e);
        vert_leaving.emplace(edge[edge[e].twin].origin, edge[e].twin);

        auto ve = add(that.vert);
        that.vert[ve].co = epos;
        edge_lut[e] = ve;
        edge_lut[edge[e].twin] = ve;
    }

    std::vector<uint32_t> vert_lut(vert.size());
    for (uint32_t v = 0; v < vert.size(); v++) {
        auto eqr = vert_leaving.equal_range(v);
        [[unlikely]] if (eqr.first == eqr.second)
            continue;

        auto vpos = [&] {
            uint32_t n = 0;
            math::vec3f vpos(0);
            for (auto it = eqr.first; it != eqr.second; ++it) {
                auto e = it->second;
                vpos += vert[edge[e].origin].co + vert[edge[edge[e].twin].origin].co;
                [[likely]] if (edge[e].face != kInvalid) {
                    vpos += that.vert[face_lut.at(edge[e].face)].co;
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

            vpos *= 1.f / n;
            vpos += (n - 3) * vert[v].co;
            vpos *= 1.f / n;
            return vpos;
        }();

        auto vv = add(that.vert);
        that.vert[vv].co = vpos;
        vert_lut[v] = vv;
    }

    std::unordered_map<uint32_t, uint32_t> tte2_lut;
    std::unordered_map<uint32_t, uint32_t> te3_lut;
    for (uint32_t f = 0; f < face_lut.size(); f++) {
        auto vf = face_lut[f];
        auto e0 = face[f].first, e = e0;

        auto lve = edge_lut.at(e);
        auto lte0 = add(that.edge);
        auto lte1 = add(that.edge);
        auto lte3 = add(that.edge);
        te3_lut.emplace(e, lte3);
        that.edge[lte0].twin = lte1;
        that.edge[lte1].twin = lte0;
        that.edge[lte0].origin = vf;
        that.edge[lte1].origin = lve;
        that.edge[lte3].origin = lve;

        auto ltf = add(that.face);
        that.face[ltf].first = lte0;
        that.edge[lte0].face = ltf;
        that.edge[lte0].next = lte3;
        that.edge[lte3].face = ltf;

        auto lte2 = add(that.edge);
        tte2_lut.emplace(edge[e].twin, lte2);
        that.edge[lte2].origin = vert_lut.at(edge[e].origin);
        that.edge[lte2].next = lte1;

        auto olte1 = lte1;
        auto olte2 = lte2;

        e = edge[e].next;

        while (e != e0) {
            auto ve = edge_lut.at(e);
            auto te0 = add(that.edge);
            auto te1 = add(that.edge);
            auto te3 = add(that.edge);
            te3_lut.emplace(e, te3);
            that.edge[te0].twin = te1;
            that.edge[te1].twin = te0;
            that.edge[te0].origin = vf;
            that.edge[te1].origin = ve;
            that.edge[te3].origin = ve;

            auto tf = add(that.face);
            that.face[tf].first = te0;
            that.edge[te0].face = tf;
            that.edge[te1].face = ltf;
            that.edge[te0].next = te3;
            that.edge[te1].next = lte0;
            that.edge[te3].face = tf;

            auto te2 = add(that.edge);
            tte2_lut.emplace(edge[e].twin, te2);
            that.edge[te2].origin = vert_lut.at(edge[e].origin);
            that.edge[te2].face = ltf;
            that.edge[te2].next = te1;
            that.edge[lte3].next = te2;

            lve = ve;
            lte0 = te0;
            lte1 = te1;
            lte3 = te3;
            ltf = tf;
            lte2 = te2;

            e = edge[e].next;
        }

        that.edge[olte1].face = ltf;
        that.edge[olte1].next = lte0;
        that.edge[olte2].face = ltf;
        that.edge[lte3].next = olte2;
    }

    for (auto const &[e, te3]: te3_lut) {
        if (auto it = tte2_lut.find(e); it != tte2_lut.end()) {
            auto te2 = it->second;
            that.edge[te2].twin = te3;
            that.edge[te3].twin = te2;
            tte2_lut.erase(it);
        } else {
            auto te2 = add(that.edge);
            that.edge[te2].next = kInvalid;
            that.edge[te2].face = kInvalid;
            that.edge[te2].twin = te3;
            that.edge[te3].twin = te2;
        }
    }

    for (auto const &[_, te2]: tte2_lut) {
        auto te3 = add(that.edge);
        that.edge[te3].next = kInvalid;
        that.edge[te3].face = kInvalid;
        that.edge[te3].twin = te2;
        that.edge[te2].twin = te3;
    }

    return that;
}


}
ZENO_NAMESPACE_END
