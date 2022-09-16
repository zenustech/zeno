#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <igl/readOFF.h>

#include <iostream>

namespace {

using namespace zeno;

struct ReadOFFPrim : zeno::INode {
    virtual void apply() override {
        auto filename = get_input2<std::string>("path");

        Eigen::MatrixXd V;
        Eigen::MatrixXi F;

        igl::readOFF(filename, V, F);

        int nm_tris = F.rows();
        int nm_vertices = V.rows();

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        prim->resize(nm_vertices);
        prim->tris.resize(nm_tris);

        for(int i = 0;i < nm_vertices;++i)
            prim->verts[i] = zeno::vec3f(V.row(i)[0],V.row(i)[1],V.row(i)[2]);
        for(int i = 0;i < nm_tris;++i)
            prim->tris[i] = zeno::vec3i(F.row(i)[0],F.row(i)[1],F.row(i)[2]);
        
        set_output("prim",std::move(prim));
    }
};

ZENDEFNODE(ReadOFFPrim, {
    {{"readpath","path"}},
    {"prim"},
    {},
    {"Parameterization"},
});

};