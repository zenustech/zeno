#include <zeno/zeno.h>
#include <zeno/types/BlenderMesh.h>
#include <zeno/types/PrimitiveObject.h>

namespace {

struct BMeshToPrimitive : zeno::INode {
    virtual void apply() override {
        auto mesh = get_input<zeno::BlenderMesh>("mesh");
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        bool allow_quads = get_param<int>("allow_quads");

        prim->resize(mesh->vert.size());
        auto &pos = prim->add_attr<zeno::vec3f>("pos");
        for (int i = 0; i < mesh->vert.size(); i++) {
            pos[i] = mesh->vert[i];
        }

        for (int i = 0; i < mesh->poly.size(); i++) {
            auto [start, len] = mesh->poly[i];
            if (len < 3) continue;
            if (len == 4 && allow_quads) {
                prim->quads.emplace_back(
                        mesh->loop[start + 0],
                        mesh->loop[start + 1],
                        mesh->loop[start + 2],
                        mesh->loop[start + 3]);
                continue;
            }
            prim->tris.emplace_back(
                    mesh->loop[start + 0],
                    mesh->loop[start + 1],
                    mesh->loop[start + 2]);
            for (int j = start + 3; j < start + len; j++) {
                prim->tris.emplace_back(
                        mesh->loop[start + 0],
                        mesh->loop[start + j - 1],
                        mesh->loop[start + j]);
            }
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(BMeshToPrimitive, {
    {"mesh"},
    {"prim"},
    {{"int", "allow_quads", "0"}},
    {"blendermesh"},
});

struct PrimitiveToBMesh : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto mesh = std::make_shared<zeno::BlenderMesh>();

        mesh->vert.resize(prim->size());
        auto &pos = prim->attr<zeno::vec3f>("pos");
        for (int i = 0; i < prim->size(); i++) {
            mesh->vert[i] = pos[i];
        }

        mesh->poly.resize(prim->tris.size() + prim->quads.size());
        mesh->loop.resize(3 * prim->tris.size() + 4 * prim->quads.size());
        for (int i = 0; i < prim->tris.size(); i++) {
            auto e = prim->tris[i];
            mesh->loop[i*3 + 0] = e[0];
            mesh->loop[i*3 + 1] = e[1];
            mesh->loop[i*3 + 2] = e[2];
            mesh->poly[i] = {i*3, 3};
        }
        int base = prim->tris.size() * 3;
        int base_poly = prim->tris.size();
        for (int i = 0; i < prim->quads.size(); i++) {
            auto e = prim->quads[i];
            mesh->loop[base + i*4 + 0] = e[0];
            mesh->loop[base + i*4 + 1] = e[1];
            mesh->loop[base + i*4 + 2] = e[2];
            mesh->loop[base + i*4 + 3] = e[3];
            mesh->poly[base_poly + i] = {base + i*4, 4};
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
