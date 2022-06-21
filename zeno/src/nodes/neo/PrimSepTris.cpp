#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>

namespace zeno {

ZENO_API void primSepTriangles(PrimitiveObject *prim, bool keepTriFaces) {
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

    prim->tris.clear();
    prim->quads.clear();
    prim->polys.clear();
    prim->loops.clear();
    prim->loop_uvs.clear();

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
    std::swap(new_verts, prim->verts);

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
        auto keepTriFaces = get_input2<bool>("keepTriFaces");
        primSepTriangles(prim.get(), keepTriFaces);
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimSepTriangles,
        { /* inputs: */ {
        {"primitive", "prim"},
        {"bool", "keepTriFaces", "1"},
        }, /* outputs: */ {
        {"primitive", "prim"},
        }, /* params: */ {
        }, /* category: */ {
        "primitive",
        }});

}
}
