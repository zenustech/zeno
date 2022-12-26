#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include "../Utils/readFile.h"
using namespace zeno;

/**
 * @brief this node just for test, do not use!
 * 
 */
struct PBFWorld_testCube2 : INode{

    virtual void apply() override{
	    auto prim = std::make_shared<zeno::PrimitiveObject>();

        //debug
        auto &pos = prim->verts;
        pos.clear();
        readVectorField("pos_init_PBFWorld.csv",pos);
        //end debug

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(PBFWorld_testCube2, {
    {},
    {"outPrim"},
    {},
    {"PBD"},
});