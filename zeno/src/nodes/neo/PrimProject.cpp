#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <unordered_map>

namespace zeno {
namespace {

static float tri_intersect(vec3f const &ro, vec3f const &rd, vec3f const &v0, vec3f const &v1, vec3f const &v2) {
        vec3f u = v1 - v0;
        vec3f v = v2 - v0;
        vec3f norm = cross(u, v);

        float b = dot(norm, rd);
        if (std::abs(b) >= eps) {
            float w0 = ro - v0;
            float a = -dot(norm, w0);
            float r = a / b;
            if (r > 0) {
                vec3f ip = ro + r * rd;
                float uu = dot(u, u);
                float uv = dot(u, v);
                float vv = dot(v, v);
                vec3f w = ip - v0;
                float wu = dot(w, u);
                float wv = dot(w, v);
                float d = uv * uv - uu * vv;
                d = 1.0f / d;
                float s = (uv * wv - vv * wu) * d;
                float t = (uv * wu - uu * wv) * d;
                if (0 <= s && s <= 1 && 0 <= t && s + t <= 1)
                    return r;
            }
    }
    return 0;
}

struct BVH {
    PrimitiveObject const *prim{};

    void build(PrimitiveObject const *prim) {
        this->prim = prim;
    }

    void intersect(vec3f ro, vec3f rd) {
        for (size_t i = 0; i < prim->tris.size(); i++) {
            auto ind = prim->tris[i];
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto c = prim->verts[ind[2]];
        }
    }
};

struct PrimProject : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto offset = get_input2<float>("offset");
        auto limit = get_input2<float>("limit");
        auto nrmAttr = get_input2<std::string>("nrmAttr");
        auto positive = get_input2<bool>("positive");
        auto negative = get_input2<bool>("negative");

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimProject, {
    {
    {"PrimitiveObject", "prim"},
    {"PrimitiveObject", "targetPrim"},
    {"string", "nrmAttr", "nrm"},
    {"float", "offset", "0"},
    {"float", "limit", "0"},
    {"bool", "positive", "1"},
    {"bool", "negitive", "1"},
    //{"bool", "targetPositive", "1"},
    //{"bool", "targetNegitive", "1"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}
}

