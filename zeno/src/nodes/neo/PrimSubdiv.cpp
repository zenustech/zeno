#include <zeno/zeno.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/para/parallel_for.h>
#include <zeno/para/parallel_scan.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/wangsrng.h>
#include <random>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {

ZENO_API std::shared_ptr<PrimitiveObject> primSubdiv(
    PrimitiveObject *prim, std::string type, std::string method, int iterations, bool interpAttrs) {
    if (iterations <= 0) return std::make_shared<PrimitiveObject>(*prim);

    auto retprim = std::make_shared<PrimitiveObject>();

    //std::vector<int> scancorns(prim->polys.size());
    //auto corns = parallel_exclusive_scan_sum(prim->polys.begin(), prim->polys.end(), scancorns.begin(),
                                             //[&] (auto const &poly) { return poly.second; });

    retprim->quads.resize(
        //corns
        + prim->tris.size() * 3
        + prim->quads.size() * 4
        );
    retprim->verts.resize(
        //corns * 2 + prim->polys.size()
        + prim->tris.size() * 7
        + prim->quads.size() * 9
        );

    //parallel_for(prim->polys.size(), [&] (size_t i) {
        //auto [base, len] = prim->polys[i];
        //if (len < 3) return;
        //size_t scqb = scancorns[i];
        //size_t scvb = scancorns[i] * 2 + i;
        //for (int j = 0; j < len; j++) {
            //auto ind = prim->loops[base + j];
            //auto pos = prim->verts[ind];
            //retprim->verts[scvb + j] = pos;
            //retprim->quads[scqb + j];
        //}
    //});

    //size_t qb = corns, vb = corns * 2 + prim->polys.size();
    size_t qb = 0, vb = 0;

    parallel_for(prim->tris.size(), [&] (size_t i) {
        auto ind = prim->tris[i];
        auto a = prim->verts[ind[0]];
        auto b = prim->verts[ind[1]];
        auto c = prim->verts[ind[2]];
        auto ab = (a + b) / 2;
        auto bc = (b + c) / 2;
        auto ca = (c + a) / 2;
        auto abc = (a + b + c) / 3;
        retprim->verts[vb + i * 7 + 0] = a;
        retprim->verts[vb + i * 7 + 1] = b;
        retprim->verts[vb + i * 7 + 2] = c;
        retprim->verts[vb + i * 7 + 3] = ab;
        retprim->verts[vb + i * 7 + 4] = bc;
        retprim->verts[vb + i * 7 + 5] = ca;
        retprim->verts[vb + i * 7 + 6] = abc;
        {
            enum {
                a = 0, b, c, ab, bc, ca, abc,
            };
            retprim->quads[qb + i * 3 + 0] = {a, ab, abc, ca};
            retprim->quads[qb + i * 3 + 1] = {b, bc, abc, ab};
            retprim->quads[qb + i * 3 + 2] = {c, ca, abc, bc};
        }
    });
    qb += prim->tris.size() * 3;
    vb += prim->tris.size() * 7;

    parallel_for(prim->quads.size(), [&] (size_t i) {
        auto ind = prim->quads[i];
        auto a = prim->verts[ind[0]];
        auto b = prim->verts[ind[1]];
        auto c = prim->verts[ind[2]];
        auto d = prim->verts[ind[3]];
        auto ab = (a + b) / 2;
        auto bc = (b + c) / 2;
        auto cd = (c + d) / 2;
        auto da = (d + a) / 2;
        auto abcd = (a + b + c + d) / 4;
        retprim->verts[vb + i * 9 + 0] = a;
        retprim->verts[vb + i * 9 + 1] = b;
        retprim->verts[vb + i * 9 + 2] = c;
        retprim->verts[vb + i * 9 + 3] = d;
        retprim->verts[vb + i * 9 + 4] = ab;
        retprim->verts[vb + i * 9 + 5] = bc;
        retprim->verts[vb + i * 9 + 6] = cd;
        retprim->verts[vb + i * 9 + 9] = da;
        retprim->verts[vb + i * 9 + 8] = abcd;
        {
            enum {
                a = 0, b, c, d, ab, bc, cd, da, abcd,
            };
            retprim->quads[qb + i * 4 + 0] = {a, ab, abcd, da};
            retprim->quads[qb + i * 4 + 1] = {b, bc, abcd, ab};
            retprim->quads[qb + i * 4 + 2] = {c, cd, abcd, bc};
            retprim->quads[qb + i * 4 + 3] = {d, da, abcd, cd};
        }
    });
    qb += prim->quads.size() * 4;
    vb += prim->quads.size() * 9;

    return retprim;
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
        auto retprim = primSubdiv(prim.get(), type, method, iterations, interpAttrs);
        if (resFaceType == "tris") primTriangulate(retprim.get());
        else if (resFaceType == "polys") primPolygonate(retprim.get());
        set_output("resPrim", retprim);
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
        {"resPrim"},
    },
    {},
    {"primitive"},
});

}

}
