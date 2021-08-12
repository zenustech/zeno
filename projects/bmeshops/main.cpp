#include <zeno/zeno.h>
#include <zeno/types/BlenderMesh.h>
#include <zeno/types/PrimitiveObject.h>

namespace {

struct BMeshToPrimitive : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<zeno::BlenderMesh>("mesh");
        auto prim = std::make_shared<zeno::PrimitiveObject>();

        prim->resize(mesh->vert.size());
        auto &pos = prim->add_attr<zeno::vec3f>("pos");
        for (int i = 0; i < mesh->vert.size(); i++) {
            pos[i] = mesh->vert[i];
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(BMeshToPrimitive, {
    {"mesh"},
    {"prim"},
    {},
    {"blendermesh"},
});

struct PrimitiveToBMesh : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto mesh = std::make_shared<zeno::BlenderMesh>();

        mesh->vert.resize(prim->size());
        auto &pos = prim->attr<zeno::vec3f>("pos");
        for (int i = 0; i < mesh->vert.size(); i++) {
            mesh->vert[i] = pos[i];
        }

        set_output("mesh", std::move(mesh));
    }
};

ZENDEFNODE(PrimitiveToBMesh, {
    {"prim"},
    {"mesh"},
    {},
    {"blendermesh"},
});

}
