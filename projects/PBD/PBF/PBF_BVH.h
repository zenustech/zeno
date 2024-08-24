#pragma once

#include <zeno/zeno.h>
#include <map>
#include <zeno/types/PrimitiveObject.h>
#include "SPHKernelFuncs.h"
#include "../ZenoFX/LinearBvh.h"

namespace zeno{
struct PBF_BVH : INode{
//physical params
public:    
    int numSubsteps = 5;
    float dt= 1.0 / 20.0;
    float pRadius = 3.0;
    vec3f bounds_max{40.0, 40.0, 40.0};
    vec3f bounds_min{0,0,0};
    vec3f externForce{0, -10.0, 0};

    float mass = 1.0;
    float rho0 = 1.0;
    float h = 1.1;
    float neighborSearchRadius = h * 1.05;
    float coeffDq = 0.3;
    float coeffK = 0.001;
    float lambdaEpsilon = 100.0; // to prevent the singularity


    void preSolve();
    void solve();
    void postSolve();

    void computeLambda();
    void computeDpos();

//Data preparing
    //data for physical fields
    int numParticles;
    std::vector<vec3f> oldPos;
    std::vector<vec3f> vel;
    std::vector<float> lambda;
    std::vector<vec3f> dpos;

    std::shared_ptr<zeno::PrimitiveObject> prim;

    //helpers
    void boundaryHandling(vec3f &p);
    inline float computeScorr(const vec3f& distVec, float coeffDq, float coeffK, float h);


    //neighborList
    std::vector<std::vector<int>> neighborList;
    std::shared_ptr<zeno::LBvh> lbvh;
    void neighborSearch(std::shared_ptr<PrimitiveObject> prim);
    void buildNeighborList(const std::vector<vec3f> &pos, float searchRadius, const zeno::LBvh *lbvh, std::vector<std::vector<int>> & list);

public:
    void setParams()
    {
        //用户自定义参数
        dt = get_input<zeno::NumericObject>("dt")->get<float>();
        pRadius = get_input<zeno::NumericObject>("particle_radius")->get<float>();
        bounds_min = get_input<zeno::NumericObject>("bounds_min")->get<zeno::vec3f>();
        bounds_max = get_input<zeno::NumericObject>("bounds_max")->get<zeno::vec3f>();
        externForce = get_input<zeno::NumericObject>("externForce")->get<zeno::vec3f>();
        rho0 = get_input<zeno::NumericObject>("rho0")->get<float>();
        h = get_input<zeno::NumericObject>("h")->get<float>();
        lambdaEpsilon = get_input<zeno::NumericObject>("lambdaEpsilon")->get<float>();
        coeffDq = get_input<zeno::NumericObject>("coeffDq")->get<float>();
        coeffK = get_input<zeno::NumericObject>("coeffK")->get<float>();

        neighborSearchRadius = h;
    }


    virtual void apply() override{
        prim = get_input<PrimitiveObject>("prim");
        auto &pos = prim->verts;

        static bool firstTime = true;
        if(firstTime == true)
        {
            firstTime = false;
            setParams();
            numParticles = prim->verts.size();

            //fields
            oldPos.resize(numParticles);
            vel.resize(numParticles);
            lambda.resize(numParticles);
            dpos.resize(numParticles);

            //构建BVH
            lbvh = std::make_shared<zeno::LBvh>(prim,  neighborSearchRadius,zeno::LBvh::element_c<zeno::LBvh::element_e::point>);
        }

        preSolve();
        neighborSearch(prim);//BVH neighborSearch
        for (size_t i = 0; i < numSubsteps; i++)
            solve(); 
        postSolve();  

        set_output("outPrim", std::move(prim));
    }
};


ZENDEFNODE(PBF_BVH, {   
                    {
                        {gParamType_Primitive, "prim"},
                        {gParamType_Vec3f, "bounds_max", "40, 40, 40"},
                        {gParamType_Vec3f, "bounds_min", "0,0,0"},
                        {gParamType_Int, "numSubsteps", "5"},
                        {gParamType_Float, "particle_radius", "3.0"},
                        {gParamType_Float, "dt", "0.05"},
                        {gParamType_Vec3f, "externForce", "0, -10, 0"},
                        {gParamType_Float, "mass", "1.0"},
                        {gParamType_Float, "rho0", "1.0"},
                        {gParamType_Float, "h", "1.1"},
                        {gParamType_Float, "coeffDq", "0.3"},
                        {gParamType_Float, "coeffK", "0.001"},
                        {gParamType_Float, "lambdaEpsilon", "100.0"}
                    },
                    {   {gParamType_Primitive, "outPrim"} },
                    {},
                    {"PBD"},
                });

}//namespace zeno