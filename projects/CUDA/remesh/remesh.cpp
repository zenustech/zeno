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
        auto iterations = get_input2<int>("iterations");
        float edge_length = edge_length = get_input2<float>("edge_length");
        auto line_pick_tag = get_input<zeno::StringObject>("line_pick_tag")->get();
        auto &pos = prim->attr<vec3f>("pos");

        auto mesh = new zeno::pmp::SurfaceMesh(prim, line_pick_tag);

    
        if (edge_length < 1e-10) {
            // If no edge length input,
            // take the average of all edges as default.
            auto &lines = prim->lines;
            for (auto eit : lines)
                edge_length += length(pos[eit[0]] - pos[eit[1]]);
            edge_length /= (double)lines.size();
            zeno::log_info("default edge_length: {}", edge_length);
        }
        zeno::pmp::SurfaceRemeshing(mesh, line_pick_tag).uniform_remeshing(
            edge_length,
            iterations);
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(UniformRemeshing)({
    {
        {"prim"},
        {"int", "iterations", "10"},
        {"float", "edge_length", "0"},
        {"string", "line_pick_tag", "line_selected"}
    },
    {"prim"},
    {},
    {"primitive"},
});

struct AdaptiveRemeshing : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto iterations = get_input2<int>("iterations");
        float max_length = get_input2<float>("max_length");
        float min_length = get_input2<float>("min_length");
        float approximation_tolerance = get_input2<float>("approximation_tolerance");
        auto line_pick_tag = get_input<zeno::StringObject>("line_pick_tag")->get();

        auto mesh = new zeno::pmp::SurfaceMesh(prim, line_pick_tag);
        auto bb = mesh->bounds().size();
        if (max_length < 1e-10) {
            max_length = 0.0500 * bb;
            zeno::log_info("default max_length: {}", max_length);
        }
        if (min_length < 1e-10) {
            min_length = 0.0010 * bb;
            zeno::log_info("default min_length: {}", min_length);
        }
        if (approximation_tolerance < 1e-10) {
            approximation_tolerance = 0.0005 * bb;
            zeno::log_info("default approximation_tolerance: {}", approximation_tolerance);
        }
        zeno::pmp::SurfaceRemeshing(mesh, line_pick_tag).adaptive_remeshing(
            min_length,
            max_length,
            approximation_tolerance,
            iterations);
        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(AdaptiveRemeshing)({
    {
        {"prim"},
        {"int", "iterations", "10"},
        {"float", "max_length", "0"},
        {"float", "min_length", "0"},
        {"float", "approximation_tolerance", "0"},
        {"string", "line_pick_tag", "line_selected"}
    },
    {"prim"},
    {},
    {"primitive"},
});
}
