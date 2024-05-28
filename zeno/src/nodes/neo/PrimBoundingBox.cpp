#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/NumericObject.h>
#include <zeno/para/parallel_reduce.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/zeno_a.h>
#include <cstring>
#include <cstdlib>

namespace zeno {

ZENO_API std::pair<vec3f, vec3f> primBoundingBox(PrimitiveObject *prim) {
    if (!prim->verts.size())
        return {{0, 0, 0}, {0, 0, 0}};
    return parallel_reduce_minmax(prim->verts.begin(), prim->verts.end());
}

namespace {

struct PrimBoundingBox : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto extraBound = get_input2<float>("extraBound");
    auto [bmin, bmax] = primBoundingBox(prim.get());
    if (extraBound != 0) {
        bmin -= extraBound;
        bmax += extraBound;
    }
    auto center = (bmin + bmax) / 2;
    auto radius = (bmax - bmin) / 2;
    auto diameter = bmax - bmin;
    set_output2("bmin", bmin);
    set_output2("bmax", bmax);
    set_output2("center", center);
    set_output2("radius", radius);
    set_output2("diameter", diameter);
  }
};

ZENO_DEFNODE(PrimBoundingBox)({
    {
    {"PrimitiveObject", "prim", "", zeno::Socket_ReadOnly},
    {"float", "extraBound", "0"},
    },
    {
    {"vec3f", "bmin"},
    {"vec3f", "bmax"},
    {"vec3f", "center"},
    {"vec3f", "radius"},
    {"vec3f", "diameter"},
    },
    {
    },
    {"primitive"},
});

ZENO_A(BoundingBoxFitInto)
ZENO_A2(vec3f, bminSrc, "0,0,0")
ZENO_A2(vec3f, bmaxSrc, "0,0,0")
ZENO_A2(vec3f, bminDst, "-1,-1,-1")
ZENO_A2(vec3f, bmaxDst, "1,1,1")
    vec3f translation(0);
    for (int i = 0; i < 3; i++) {
        bool lt = bminSrc[i] < bminDst[i], gt = bmaxSrc[i] > bmaxDst[i];
        if (lt && !gt) {
            translation[i] = bminDst[i] - bminSrc[i];
        } else if (!lt && gt) {
            translation[i] = bmaxDst[i] - bmaxSrc[i];
        } else if (lt && gt) {
            translation[i] = (bminDst[i] - bminSrc[i] + bmaxDst[i] - bmaxSrc[i]) / 2;
        }
    }

    auto bminTrans = bminSrc + translation;
    auto bmaxTrans = bmaxSrc + translation;
ZENO_B2(vec3f, translation)
ZENO_B2(vec3f, bminTrans)
ZENO_B2(vec3f, bmaxTrans)
ZENO_C()

_ZENO_A(BoundingBoxFitInto)
_ZENO_A2(vec3f, bminSrc, "0,0,0")
_ZENO_A2(vec3f, bmaxSrc, "0,0,0")
_ZENO_A2(vec3f, bminDst, "-1,-1,-1")
_ZENO_A2(vec3f, bmaxDst, "1,1,1")
_ZENO_B()
_ZENO_B2(vec3f, translation)
_ZENO_B2(vec3f, bminTrans)
_ZENO_B2(vec3f, bmaxTrans)
_ZENO_C(primitive)

struct PrimCalcCentroid : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto method = get_input2<std::string>("method");
        auto density = get_input2<float>("density");

        vec4f acc;
        if (method == "Vertex") {
            acc = parallel_reduce_sum(prim->verts.begin(), prim->verts.end(), [&] (vec3f const &pos) {
                return vec4f(pos[0], pos[1], pos[2], 1.0f);
            });

        } else if (method == "Area") {
            acc = parallel_reduce_sum(prim->tris.begin(), prim->tris.end(), [&] (vec3i const &ind) {
                auto a = prim->verts[ind[0]], b = prim->verts[ind[1]], c = prim->verts[ind[2]];
                auto weight = length(cross(b - a, c - a));
                auto center = weight * (1.f / 3.f) * (a + b + c);
                return vec4f(center[0], center[1], center[2], weight);
            });

        } else if (method == "BoundBox") {
            auto [bmin, bmax] = primBoundingBox(prim.get());
            auto center = (bmin + bmax) / 2;
            auto diam = bmax - bmin;
            auto volume = diam[0] * diam[1] * diam[2];
            center *= volume;
            acc = vec4f(center[0], center[1], center[2], volume);

        } else {
            acc = parallel_reduce_sum(prim->tris.begin(), prim->tris.end(), [&] (vec3i const &ind) {
                auto a = prim->verts[ind[0]], b = prim->verts[ind[1]], c = prim->verts[ind[2]];
                auto normal = cross(b - a, c - a);
                auto area = length(normal); normal /= area;
                auto center = (1.f / 3.f) * (a + b + c);
                auto weight = (1.f / 6.f) * area * dot(center, normal);
                center = (3.f / 4.f) * weight * center;
                return vec4f(center[0], center[1], center[2], weight);
            });
        }

        auto centroid = acc[3] == 0 ? vec3f(0) :
            vec3f(acc[0], acc[1], acc[2]) / acc[3];
        auto mass = std::abs(acc[3]) * density;

        set_output2("centroid", centroid);
        set_output2("mass", mass);
    }
};

ZENO_DEFNODE(PrimCalcCentroid)({
    {
    {"PrimitiveObject", "prim", "", zeno::Socket_ReadOnly},
    {"enum Volume Area Vertex BoundBox", "method", "Volume"},
    {"float", "density", "1"},
    },
    {
    {"vec3f", "centroid"},
    {"float", "mass"},
    },
    {
    },
    {"primitive"},
});

}
}
