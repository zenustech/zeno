#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
//#include <zeno/para/execution.h>
//#include <algorithm>

namespace zeno {

#if 0
ZENO_API void primSmoothNormal(PrimitiveObject *prim) {
    auto &nrm = prim->verts.add_attr<vec3f>("nrm");
    std::fill(ZENO_PAR_UNSEQ nrm.begin(), nrm.end(), vec3f());
    for (size_t i = 0; i < prim->polys.size(); i++) {
        auto [base, len] = prim->polys[i];
        if (len < 3) continue;
        auto a = prim->verts[prim->loops[base]];
        auto b = prim->verts[prim->loops[base + 1]];
        auto c = prim->verts[prim->loops[base + 2]];
        auto n = cross(b - a, c - a);
        for (size_t j = base + 2; j < base + len; j++) {
            auto b = prim->verts[prim->loops[j]];
            auto c = prim->verts[prim->loops[j + 1]];
            n += cross(b - a, c - a);
        }
        for (size_t j = base; j < base + len; j++) {
            nrm[j] += n;
        }
    }
    std::for_each(ZENO_PAR_UNSEQ nrm.begin(), nrm.end(), [] (vec3f &n) {
        n = normalizeSafe(n, 1e-6f);
    });
}
#endif

ZENO_API void primSepTriangles(PrimitiveObject *prim, bool smoothNormal, bool keepTriFaces) {
    if (!prim->tris.size() && !prim->quads.size() && !prim->polys.size()) {
        //if ((prim->points.size() || prim->lines.size()) && !prim->verts.has_attr("clr")) {
            //throw;
            //prim->verts.add_attr<vec3f>("clr").assign(prim->verts.size(), vec3f(1, 1, 0));
        //}
        return; // TODO: cihou pars and lines
    }
    // TODO: support index compress?
    //if (smoothNormal && !prim->verts.has_attr("nrm")) {
        //primSmoothNormal(prim);
    //}
    bool needCompNormal = !prim->verts.has_attr("nrm");
    bool needCompUVs = !prim->verts.has_attr("uv");

    std::vector<int> v;
    int loopcount = 0;
    for (size_t i = 0; i < prim->polys.size(); i++) {
        auto [base, len] = prim->polys[i];
        if (len < 3) continue;
        loopcount += len - 2;
    }
    v.resize(prim->tris.size() * 3 + prim->quads.size() * 6 + loopcount * 3);
    for (size_t i = 0; i < prim->tris.size(); i++) {
        auto ind = prim->tris[i];
        v[i * 3 + 0] = ind[0];
        v[i * 3 + 1] = ind[1];
        v[i * 3 + 2] = ind[2];
    }
    size_t b = prim->tris.size() * 3;
    for (size_t i = 0; i < prim->quads.size(); i++) {
        auto ind = prim->quads[i];
        v[b + i * 6 + 0] = ind[0];
        v[b + i * 6 + 1] = ind[1];
        v[b + i * 6 + 2] = ind[2];
        v[b + i * 6 + 3] = ind[0];
        v[b + i * 6 + 4] = ind[2];
        v[b + i * 6 + 5] = ind[3];
    }
    b += prim->quads.size() * 6;
    for (size_t i = 0; i < prim->polys.size(); i++) {
        auto [base, len] = prim->polys[i];
        if (len < 3) continue;
        v[b] = prim->loops[base];
        v[b + 1] = prim->loops[base + 1];
        v[b + 2] = prim->loops[base + 2];
        for (int j = 0; j < len - 3; j++) {
            v[b + 3 + 3 * j] = prim->loops[base];
            v[b + 4 + 3 * j] = prim->loops[base + j + 2];
            v[b + 5 + 3 * j] = prim->loops[base + j + 3];
        }
        b += (len - 2) * 3;
    }

    AttrVector<vec3f> new_verts;
    new_verts.resize(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        new_verts[i] = prim->verts[v[i]];
    }
    prim->verts.foreach_attr([&] (auto const &key, auto const &arr) {
        using T = std::decay_t<decltype(arr[0])>;
        auto &new_arr = new_verts.add_attr<T>(key);
        for (size_t i = 0; i < v.size(); i++) {
            new_arr[i] = arr[v[i]];
        }
    });

    if (needCompUVs) {
        if (prim->tris.has_attr("uv0") &&
            prim->tris.has_attr("uv1") &&
            prim->tris.has_attr("uv2")) {
            auto &uv0 = prim->tris.attr<vec3f>("uv0");
            auto &uv1 = prim->tris.attr<vec3f>("uv1");
            auto &uv2 = prim->tris.attr<vec3f>("uv2");
            auto &new_uv = new_verts.add_attr<vec3f>("uv");
            for (int i = 0; i < prim->tris.size(); i++) {
                auto uv = uv0[i];
                new_uv[i * 3 + 0] = {uv[0], uv[1], 0};
                uv = uv1[i];
                new_uv[i * 3 + 1] = {uv[0], uv[1], 0};
                uv = uv2[i];
                new_uv[i * 3 + 2] = {uv[0], uv[1], 0};
            }
        }
        if (prim->quads.has_attr("uv0") &&
            prim->quads.has_attr("uv1") &&
            prim->quads.has_attr("uv2") &&
            prim->quads.has_attr("uv3")) {
            auto &uv0 = prim->quads.attr<vec3f>("uv0");
            auto &uv1 = prim->quads.attr<vec3f>("uv1");
            auto &uv2 = prim->quads.attr<vec3f>("uv2");
            auto &uv3 = prim->quads.attr<vec3f>("uv3");
            auto &new_uv = new_verts.add_attr<vec3f>("uv");
            size_t b = prim->tris.size() * 3;
            for (int i = 0; i < prim->quads.size(); i++) {
                new_uv[b + i * 6 + 0] = uv0[i];
                new_uv[b + i * 6 + 1] = uv1[i];
                new_uv[b + i * 6 + 2] = uv2[i];
                new_uv[b + i * 6 + 3] = uv0[i];
                new_uv[b + i * 6 + 4] = uv2[i];
                new_uv[b + i * 6 + 5] = uv3[i];
            }
        }
        if (prim->loop_uvs.size()) {
            size_t b = 0;
            std::vector<int> v(loopcount * 3);
            for (size_t i = 0; i < prim->polys.size(); i++) {
                auto [base, len] = prim->polys[i];
                if (len < 3) continue;
                v[b] = prim->loop_uvs[base];
                v[b + 1] = prim->loop_uvs[base + 1];
                v[b + 2] = prim->loop_uvs[base + 2];
                for (int j = 0; j < len - 3; j++) {
                    v[b + 3 + 3 * j] = prim->loop_uvs[base];
                    v[b + 4 + 3 * j] = prim->loop_uvs[base + j + 2];
                    v[b + 5 + 3 * j] = prim->loop_uvs[base + j + 3];
                }
                b += (len - 2) * 3;
            }
            b = prim->tris.size() * 3 + prim->quads.size() * 6;
            auto &new_uv = new_verts.add_attr<vec3f>("uv");
            for (int i = 0; i < v.size(); i++) {
                auto uv = prim->uvs[v[i]];
                new_uv[b + i] = {uv[0], uv[1], 0};
            }
        }
    }

    prim->tris.clear();
    prim->quads.clear();
    prim->polys.clear();
    prim->loops.clear();
    prim->loop_uvs.clear();
    prim->uvs.clear();

    if (smoothNormal && needCompNormal) {
        std::vector<vec3f> shn(prim->verts.size());
        for (size_t i = 0; i < v.size() / 3; i++) {
            auto a = new_verts[i * 3 + 0];
            auto b = new_verts[i * 3 + 1];
            auto c = new_verts[i * 3 + 2];
            auto n = cross(b - a, c - a);
            n = normalizeSafe(n, 1e-6f);
            shn[v[i * 3 + 0]] += n;
            shn[v[i * 3 + 1]] += n;
            shn[v[i * 3 + 2]] += n;
        }
        auto &new_nrm = new_verts.add_attr<vec3f>("nrm");
        for (size_t i = 0; i < v.size(); i++) {
            auto n = shn[v[i]];
            n = normalizeSafe(n, 1e-6f);
            new_nrm[i] = n;
        }
    }

    std::swap(new_verts, prim->verts);

    if (!smoothNormal && needCompNormal) {
        auto &nrm = prim->verts.add_attr<vec3f>("nrm");
        for (size_t i = 0; i < v.size() / 3; i++) {
            auto a = prim->verts[i * 3 + 0];
            auto b = prim->verts[i * 3 + 1];
            auto c = prim->verts[i * 3 + 2];
            auto n = cross(b - a, c - a);
            n = normalizeSafe(n, 1e-6f);
            nrm[i * 3 + 0] = n;
            nrm[i * 3 + 1] = n;
            nrm[i * 3 + 2] = n;
        }
    }

    if (keepTriFaces) {
        prim->tris.resize(v.size() / 3);
        for (int i = 0; i < prim->tris.size(); i++) {
            prim->tris[i] = {i * 3, i * 3 + 1, i * 3 + 2};
        }
    }
}

namespace {

struct PrimSepTriangles : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto smoothNormal = get_input2<bool>("smoothNormal");
        auto keepTriFaces = get_input2<bool>("keepTriFaces");
        //printf("asdasd %d\n", prim->verts.attrs.erase("nrm"));
        primSepTriangles(prim.get(), smoothNormal, keepTriFaces);
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimSepTriangles,
        { /* inputs: */ {
        {"primitive", "prim"},
        {"bool", "smoothNormal", "1"},
        {"bool", "keepTriFaces", "1"},
        }, /* outputs: */ {
        {"primitive", "prim"},
        }, /* params: */ {
        }, /* category: */ {
        "primitive",
        }});

}
}
