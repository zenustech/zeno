#include <zeno/zeno.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/para/parallel_for.h>
#include <zeno/para/parallel_scan.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/wangsrng.h>
#include <zeno/utils/tuple_hash.h>
#include <unordered_map>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {

ZENO_API void primSubdiv(PrimitiveObject *prim, std::string type, std::string method, int iterations, bool interpAttrs) {
    if (iterations <= 0) return;

    auto sorted = [] (int i, int j) {
        return std::make_pair(std::min(i, j), std::max(i, j));
    };

    std::vector<vec3f> tricents(prim->tris.size());
    std::unordered_map<std::pair<int, int>, std::vector<int>, tuple_hash, tuple_equal> edgelut;
    edgelut.reserve(prim->tris.size());
    for (size_t i = 0; i < prim->tris.size(); i++) {
        auto ind = prim->tris[i];
        auto a = prim->verts[ind[0]];
        auto b = prim->verts[ind[1]];
        auto c = prim->verts[ind[2]];
        tricents[i] = (a + b + c) / 3;
        edgelut[sorted(ind[0], ind[1])].push_back(i);
        edgelut[sorted(ind[1], ind[2])].push_back(i);
        edgelut[sorted(ind[2], ind[0])].push_back(i);
    }

    std::vector<vec3f> edgecents(edgelut.size());
    std::vector<std::array<int, 4>> trineiedges(tricents.size());
    size_t edgecentscnt = 0;
    for (auto const &[edge, triids]: edgelut) {
        auto a = prim->verts[edge.first];
        auto b = prim->verts[edge.second];
        auto id = edgecentscnt++;
        for (auto triid: triids) {
            auto &neiarr = trineiedges[triid];
            neiarr[++neiarr[0]] = id;
        }
        edgecents[id] = (a + b) / 2;
    }

    const size_t oldnumverts = prim->verts.size();
    prim->verts.resize(oldnumverts + edgecents.size() + tricents.size());

    for (size_t i = 0; i < edgecents.size(); i++) {
        prim->verts[oldnumverts + i] = edgecents[i];
    }

    for (size_t i = 0; i < tricents.size(); i++) {
        prim->verts[oldnumverts + edgecents.size() + i] = tricents[i];
    }

    prim->tris.clear();
    prim->quads.clear();
    prim->quads.resize(tricents.size());

    for (size_t i = 0; i < tricents.size(); i++) {
        auto ind = prim->tris[i];
        int i01 = oldnumverts + i;
        int i20 = oldnumverts + i;
        int i012 = oldnumverts + edgecents.size() + i;
        trineiedges[i][1];
        trineiedges[i][2];
        trineiedges[i][3];
        prim->quads[i] = {ind[0], i01, i012, i20};
    }
}

namespace {

struct PrimSubdiv : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto type = get_input2<std::string>("type");
        auto method = get_input2<std::string>("method");
        auto iterations = get_input2<int>("iterations");
        auto interpAttrs = get_input2<bool>("interpAttrs");
        auto resFaceType = get_input2<std::string>("resFaceType");
        primSubdiv(prim.get(), type, method, iterations, interpAttrs);
        if (resFaceType == "tris") primTriangulate(prim.get());
        else if (resFaceType == "polys") primPolygonate(prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(PrimSubdiv)({
    {
        {"prim"},
        {"enum faces lines", "type", "faces"},
        {"enum catmull simple", "method", "catmull"},
        {"int", "iterations", "1"},
        {"bool", "interpAttrs", "1"},
        {"enum tris quads polys", "resFaceType", "tris"},
    },
    {
        {"prim"},
    },
    {},
    {"primitive"},
});

}

}
