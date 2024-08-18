#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include "./PBFWorld.h"
#include "../Utils/myPrint.h"//debug
#include "../Utils/readFile.h"//debug
using namespace zeno;

struct PBFWorld_Debug : zeno::INode {

     virtual void apply() override{
        auto prim = get_input<PrimitiveObject>("prim");
        auto data = get_input<PBFWorld>("PBFWorld");

        //绑定PBFWorld中的field到attr
        auto &prim_vel = prim->add_attr<vec3f>("vel");
        auto &prim_lambda = prim->add_attr<float>("lambda");
        auto &prim_dpos = prim->add_attr<vec3f>("dpos");
        auto &prim_prevPos = prim->add_attr<vec3f>("prevPos");
        prim_vel = data->vel;
        prim_lambda = data->lambda;
        prim_dpos = data->dpos;
        prim_prevPos = data->prevPos;

        //绑定参数到userData
        prim->userData().set("dt", std::make_shared<NumericObject>(data->dt));
        prim->userData().set("radius", std::make_shared<NumericObject>(data->radius));
        prim->userData().set("rho0", std::make_shared<NumericObject>(data->rho0));
        prim->userData().set("mass", std::make_shared<NumericObject>(data->mass));
        prim->userData().set("numParticles", std::make_shared<NumericObject>(data->numParticles));
        prim->userData().set("h", std::make_shared<NumericObject>(data->h));


        set_output("outPrim", std::move(prim));
        set_output("PBFWorld", std::move(data));
    }
};

ZENDEFNODE(
    PBFWorld_Debug,
    {
        {gParamType_Primitive,"PBFWorld"},
        {"outPrim","PBFWorld"},
        {},
        {"PBD"}
    }
);