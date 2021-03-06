#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/para/parallel_for.h>  // enable by -DZENO_PARALLEL_STL:BOOL=ON
#include <unordered_map>
#include <functional>
#include <limits>

namespace zeno {
namespace {

template <class Cond>
static float tri_intersect(Cond cond, vec3f const &ro, vec3f const &rd,
                           vec3f const &v0, vec3f const &v1, vec3f const &v2) {
    const float eps = 1e-6f;
    vec3f u = v1 - v0;
    vec3f v = v2 - v0;
    vec3f n = cross(u, v);
    float b = dot(n, rd);
    if (std::abs(b) > eps) {
        float a = dot(n, v0 - ro);
        float r = a / b;
        if (cond(r)) {
            vec3f ip = ro + r * rd;
            float uu = dot(u, u);
            float uv = dot(u, v);
            float vv = dot(v, v);
            vec3f w = ip - v0;
            float wu = dot(w, u);
            float wv = dot(w, v);
            float d = uv * uv - uu * vv;
            float s = uv * wv - vv * wu;
            float t = uv * wu - uu * wv;
            d = 1.0f / d;
            s *= d;
            t *= d;
            if (-eps <= s && s <= 1 + eps && -eps <= t && s + t <= 1 + eps * 2)
                return r;
        }
    }
    return std::numeric_limits<float>::infinity();
}

struct BVH {  // TODO: WXL please complete this to accel up
    PrimitiveObject const *prim{};

    void build(PrimitiveObject const *prim) {
        this->prim = prim;
    }

    template <class Cond>
    float intersect(Cond cond, vec3f const &ro, vec3f const &rd) const {
        float ret = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < prim->tris.size(); i++) {
            auto ind = prim->tris[i];
            auto a = prim->verts[ind[0]];
            auto b = prim->verts[ind[1]];
            auto c = prim->verts[ind[2]];
            float d = tri_intersect(cond, ro, rd, a, b, c);
            if (std::abs(d) < std::abs(ret))
                ret = d;
        }
        return ret;
    }
};

struct PrimProject : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto targetPrim = get_input<PrimitiveObject>("targetPrim");
        auto offset = get_input2<float>("offset");
        auto limit = get_input2<float>("limit");
        auto nrmAttr = get_input2<std::string>("nrmAttr");
        auto allowDir = get_input2<std::string>("allowDir");

        BVH bvh;
        bvh.build(targetPrim.get());

        if (limit <= 0)
            limit = std::numeric_limits<float>::infinity();

        struct allow_front {
            bool operator()(float x) const {
                return x >= 0;
            }
        };

        struct allow_back {
            bool operator()(float x) const {
                return x <= 0;
            }
        };

        struct allow_both {
            bool operator()(float x) const {
                return true;
            }
        };

        auto const &nrm = prim->verts.attr<vec3f>(nrmAttr);
        auto cond = enum_variant<std::variant<
            allow_front, allow_back, allow_both
            >>(array_index({"front", "back", "both"}, allowDir));

        std::visit([&] (auto cond) {
            parallel_for((size_t)0, prim->verts.size(), [&] (size_t i) {
                auto ro = prim->verts[i];
                auto rd = normalizeSafe(nrm[i]);
                float t = bvh.intersect(cond, ro, rd);
                if (std::abs(t) >= limit)
                    t = 0;
                t -= offset;
                prim->verts[i] = ro + t * rd;
            });
        }, cond);

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
    {"enum front back both", "allowDir", "both"},
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

