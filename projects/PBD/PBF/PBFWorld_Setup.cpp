#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include "./PBFWorld.h"
#include "../Utils/myPrint.h"//debug
#include "../Utils/readFile.h"//debug
#include "./SPHKernels.h"//debug

using namespace zeno;

/**
 * @brief 接受用户输入参数并设置数据类型PBFWorld
 * 
 */
struct PBFWorld_Setup : INode 
{
    void apply() override 
    {
        //创建数据类，用data指针来记录
        auto data = std::make_shared<PBFWorld>();

        auto prim = get_input<PrimitiveObject>("prim");
        
        //设置物理场
        data->numParticles = prim->verts.size();
        data->vel.resize(data->numParticles);
        data->prevPos.resize(data->numParticles);
        data->lambda.resize(data->numParticles);
        data->dpos.resize(data->numParticles);

        //用户自定义参数
        data->dt = get_input<zeno::NumericObject>("dt")->get<float>();
        data->radius = get_input<zeno::NumericObject>("particle_radius")->get<float>();
        data->bounds_min = get_input<zeno::NumericObject>("bounds_min")->get<vec3f>();
        data->bounds_max = get_input<zeno::NumericObject>("bounds_max")->get<vec3f>();
        data->externForce = get_input<zeno::NumericObject>("externForce")->get<vec3f>();
        data->rho0 = get_input<zeno::NumericObject>("rho0")->get<float>();
        data->lambdaEpsilon = get_input<zeno::NumericObject>("lambdaEpsilon")->get<float>();
        data->coeffDq = get_input<zeno::NumericObject>("coeffDq")->get<float>();
        data->coeffK = get_input<zeno::NumericObject>("coeffK")->get<float>();

        //可以推导出来的参数
        auto diam = data->radius*2;
        data->mass = 0.8 * diam*diam*diam * data->rho0;
        data->h = 4* data->radius;
        data->neighborSearchRadius = data->h;

        //初始化Kernel
        // CubicKernel::set(data->h);

        //传出数据
        set_output("prim", std::move(prim));
        set_output("PBFWorld", std::move(data));
    }
};


ZENDEFNODE(PBFWorld_Setup, {
    {
        {"PrimitiveObject","prim"},
        {"float","dt"," 0.0025"},
        {"float","particle_radius","0.025"},
        {"vec3f","bounds_min","-10.0, 0.0, -10.0"},
        {"vec3f","bounds_max","10.0, 10.0, 10.0"},
        {"vec3f","externForce", "0.0, -10.0, 0.0"},
        {"float","rho0","1000.0"},
        {"float","lambdaEpsilon","1e-6"},
        {"float","coeffDq","0.3"},
        {"float","coeffK","0.1"},
        {"int","numSubsteps","5"}
    },
    {"prim","PBFWorld"},
    {},
    {"PBD"},
});
