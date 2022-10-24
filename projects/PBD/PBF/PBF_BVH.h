#pragma once

#include <zeno/zeno.h>
#include <map>
#include <zeno/types/PrimitiveObject.h>
#include "SPHKernelFuncs.h"

namespace zeno{
struct PBF_BVH : INode{
//physical params
public:    
    int numSubsteps = 5;
    float dt= 1.0 / 20.0;
    float pRadius = 3.0;
    vec3f bounds_max{40.0, 40.0, 40.0};
    vec3f bounds_min{0,0,0};
    vec3f gravity{0, -10.0, 0};

    float mass = 1.0;
    float rho0 = 1.0;
    float h = 1.1;
    float neighborSearchRadius = h * 1.05;
    float coeffDq = 0.3;
    float coeffK = 0.001;
    float lambdaEpsilon = 100.0; // to prevent the singularity

private:
    void preSolve();
    void solve();
    void postSolve();

    void computeLambda();
    void computeDpos();

    void PBF_BVH::boundaryHandling(vec3f & p);

    float computeScorr(const vec3f& distVec, float coeffDq, float coeffK, float h);

    void PBF_BVH::neighborhoodSearch(std::shared_ptr<PrimitiveObject> prim);
    void PBF_BVH::buildNeighborList(const std::vector<vec3f> &pos, float searchRadius, const zeno::LBvh *lbvh, std::vector<std::vector<int>> & list);

//Data preparing
    //data for physical fields
    int numParticles;
    // std::vector<vec3f> pos;
    std::vector<vec3f> oldPos;
    std::vector<vec3f> vel;
    std::vector<float> lambda;
    std::vector<vec3f> dpos;

    std::shared_ptr<zeno::PrimitiveObject> prim;

    //helpers
    void boundaryHandling(vec3f &p);
    inline float computeScorr(const vec3f& distVec, float coeffDq, float coeffK, float h);

    // void initNeighborList();

    //neighborList
    std::vector<std::vector<int>> neighborList;
    void neighborSearch();

public:
    void setParams()
    {
        //用户自定义参数
        dt = get_input<zeno::NumericObject>("dt")->get<float>();
        pRadius = get_input<zeno::NumericObject>("particle_radius")->get<float>();
        bounds_min = get_input<zeno::NumericObject>("bounds_min")->get<vec3f>();
        bounds_max = get_input<zeno::NumericObject>("bounds_max")->get<vec3f>();
        gravity = get_input<zeno::NumericObject>("gravity")->get<vec3f>();
        rho0 = get_input<zeno::NumericObject>("rho0")->get<float>();
        lambdaEpsilon = get_input<zeno::NumericObject>("lambdaEpsilon")->get<float>();
        coeffDq = get_input<zeno::NumericObject>("coeffDq")->get<float>();
        coeffK = get_input<zeno::NumericObject>("coeffK")->get<float>();

        
        //可以推导出来的参数
        // auto diam = pRadius*2;
        // mass = 0.8 * diam*diam*diam * rho0;
        // h = 4* pRadius;
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

            // initCellData();
            // initNeighborList(); 
        }

        preSolve();
        neighborSearch();//grid-baed neighborSearch
        for (size_t i = 0; i < numSubsteps; i++)
            solve(); 
        postSolve();  

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(PBF_BVH, {   
                    {
                        {"PrimitiveObject", "prim"},
                        {"vec3f", "bounds_max", "40, 40, 40"},
                        {"vec3f", "bounds_min", "0,0,0"},
                        {"int", "numSubsteps", "5"},
                        {"float", "particle_radius", "3.0"},
                        {"float", "dt", "0.05"},
                        {"vec3f", "gravity", "0, -10, 0"},
                        {"float", "mass", "1.0"},
                        {"float", "rho0", "1.0"},
                        {"float", "coeffDq", "0.3"},
                        {"float", "coeffK", "0.001"},
                        {"float", "lambdaEpsilon", "100.0"}
                    },
                    {   {"PrimitiveObject", "outPrim"} },
                    {},
                    {"PBD"},
                });
}//namespace zeno