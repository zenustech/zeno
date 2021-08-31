#include <zeno/zeno.h>
#include <zeno/utils/vec.h>
#include <zeno/types/PrimitiveObject.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <Eigen/Core>

namespace {
using namespace zeno;


struct PrimitiveBooleanOp : INode {
    virtual void apply() override {
        auto prim1 = get_input<PrimitiveObject>("prim1");
        auto prim2 = get_input<PrimitiveObject>("prim2");
        auto op_type = get_param<std::string>("op_type");

        igl::copyleft::cgal::mesh_boolean(VA,FA,VB,FB,boolean_type,VC,FC,J);

        auto res = std::make_shared<PrimitiveObject>();
        auto &pos = res->add_attr<vec3f>("pos");
        for (auto vit = result.vertices_begin(); vit != result.vertices_end(); vit++) {
            float x = vit->point().x().exact().convert_to<float>();
            float y = vit->point().y().exact().convert_to<float>();
            float z = vit->point().z().exact().convert_to<float>();
            pos.emplace_back(x, y, z);
        }
        res->resize(pos.size());
        for (auto fsit = result.facets_begin(); fsit != result.facets_end(); fsit++) {
            int fitid = 0;
            for (auto fit = fsit->facet_begin(); fitid++ < fsit->facet_degree(); fit++) {
                printf("%zd\n", fit->vertex_degree());
            }
        }

        set_output("prim", std::move(res));
    }
};

ZENO_DEFNODE(PrimitiveBooleanOp)({
    {
    "prim1", "prim2"
    },
    {
    "prim",
    },
    {
    {"enum union intersection difference", "op_type", "union"},
    },
    {"cgmesh"},
});

}
