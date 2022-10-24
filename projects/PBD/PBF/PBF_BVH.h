#pragma once

#include <zeno/zeno.h>
#include <map>
#include <zeno/types/PrimitiveObject.h>
#include "SPHKernelFuncs.h"
#include "../ZenoFX/LinearBvh.h" //BVH搜索
#include "PBFWorld.h" 
#include "../Utils/myPrint.h" 

namespace zeno{
struct PBF_BVH : INode{
//physical params
public:    
    int numSubsteps = 5;
    float dt= 1.0 / 20.0;
    float pRadius = 3.0;
    // vec3f bounds{40.0, 40.0, 40.0};
    vec3f bounds_min{0,0,0};
    vec3f bounds_max{40.0, 40.0, 40.0};
    vec3f g{0, -10.0, 0};

    float mass = 1.0;
    float rho0 = 1.0;
    float h = 1.1;
    float neighborSearchRadius = h * 1.05;
    float coeffDq = 0.3;
    float coeffK = 0.001;

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
    std::vector<vec3f> pos;
    std::vector<vec3f> oldPos;
    std::vector<vec3f> vel;
    std::vector<float> lambda;
    std::vector<vec3f> dpos;

    std::shared_ptr<zeno::PrimitiveObject> prim;

    std::vector<std::vector<int>> neighborList;
    std::shared_ptr<zeno::LBvh> lbvh;


    virtual void apply() override{
        prim = get_input<PrimitiveObject>("prim");
        // auto data = get_input<PBFWorld>("PBFWorld");

        static bool firstTime = true;
        if(firstTime == true)
        {
            firstTime = false;

            // move pos to local
            numParticles = prim->verts.size();
            pos = std::move(prim->verts);

            //prepare physical field data
            oldPos.resize(numParticles);
            vel.resize(numParticles);
            lambda.resize(numParticles);
            dpos.resize(numParticles);

            //构建BVH
            lbvh = std::make_shared<zeno::LBvh>(prim,  neighborSearchRadius,zeno::LBvh::element_c<zeno::LBvh::element_e::point>);
        }


        preSolve();
        // std::cout<<"good\n";
        for (size_t i = 0; i < numSubsteps; i++)
            solve(); 
        postSolve();  

        prim->verts.resize(pos.size());
        for (size_t i = 0; i < pos.size(); i++)
            prim->verts[i] = pos[i]/10.0;//scale to show

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(PBF_BVH, {   
                    {
                        // {"PBFWorld"},
                        {"PrimitiveObject", "prim"},
                        {"vec3f", "bounds_max", "40, 40, 40"}
                        // {"int", "numSubsteps", "5"}
                    },
                    {   {"PrimitiveObject", "outPrim"} },
                    {},
                    {"PBD"},
                });
}//namespace zeno