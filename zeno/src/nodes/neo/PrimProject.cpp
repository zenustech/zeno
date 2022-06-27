#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <unordered_map>

namespace zeno {
namespace {

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

