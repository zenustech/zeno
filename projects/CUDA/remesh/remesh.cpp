#include <zeno/zeno.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <filesystem>
#include <sstream>
#include <cctype>


#include "./SurfaceMesh.h"
#include "./algorithms/SurfaceRemeshing.h"

namespace zeno {

struct UniformRemeshing : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto iterations = get_param<int>("iterations");
        auto &pos = prim->attr<vec3f>("pos");

        auto mesh = new zeno::pmp::SurfaceMesh(prim);

        auto &edges = prim->edges;
        double l(0);
        for (auto eit : edges)
            l += length(pos[eit[0]] - pos[eit[1]]);
        l /= (double)edges.size();
        zeno::pmp::SurfaceRemeshing(mesh).uniform_remeshing(l, iterations);
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(UniformRemeshing)({
    {
        {"prim"},
    },
    {"prim"},
    {
        {"int", "iterations", "10"},
    },
    {"primitive"},
});

struct AdaptiveRemeshing : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto iterations = get_param<int>("iterations");

        auto mesh = new zeno::pmp::SurfaceMesh(prim);
        auto bb = mesh->bounds().size();
        zeno::pmp::SurfaceRemeshing(mesh).adaptive_remeshing(
            0.0010 * bb,  // min length
            0.0500 * bb,  // max length
            0.0005 * bb,  // approx. error
            iterations);
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(AdaptiveRemeshing)({
    {
        {"prim"},
    },
    {"prim"},
    {
        {"int", "iterations", "10"},
    },
    {"primitive"},
});
}
