#pragma once

#include <zeno/zeno.h>
#include <map>
#include <zeno/types/PrimitiveObject.h>
#include "SPHKernelFuncs.h"

namespace zeno{
struct PBF : INode{
//physical params
public:    
    int numSubsteps = 5;
    float dt= 1.0 / 20.0;
    float pRadius = 3.0;
    vec3f bounds{40.0, 40.0, 40.0};
    vec3f g{0, -10.0, 0};

    float mass = 1.0;
    float rho0 = 1.0;
    float h = 1.1;
    float neighborSearchRadius = h * 1.05;

private:
    void preSolve();
    void solve();
    void postSolve();

    void computeLambda();
    void computeDpos();
    void neighborSearch();

//Data preparing
    //data for physical fields
    void readPoints();
    void initData();
    int numParticles;
    std::vector<vec3f> pos;
    std::vector<vec3f> oldPos;
    std::vector<vec3f> vel;
    std::vector<float> lambda;
    std::vector<vec3f> dpos;

    std::shared_ptr<zeno::PrimitiveObject> prim;

    //helpers
    void boundaryHandling(vec3f &p);
    inline vec3i getCellXYZ(const vec3f& p);
    inline int getCellID(const vec3f& p);
    inline int getCellHash(int i, int j, int k);
    inline bool isInBound(const vec3i& cell);
    inline int cellXYZ2ID(const vec3i& xyz);
    inline vec3i cellID2XYZ(int i);
    inline float computeScorr(const vec3f& distVec);

    //data for cells
    std::vector<int> numCellXYZ;
    int numCell;
    float dx; //cell size
    float dxInv; 
    vec3i bound;
    void initCellData();
    struct Cell
    {
        int x,y,z;
        std::vector<int> parInCell; 
    };
    std::map<int, Cell>  cell;

    //neighborList
    std::vector<std::vector<int>> neighborList;

public:
    virtual void apply() override{
        prim = get_input<PrimitiveObject>("prim");

        static bool firstTime = true;
        if(firstTime == true)
        {
            firstTime = false;

            initCellData();

            // move pos to local
            numParticles = prim->verts.size();
            pos = std::move(prim->verts);

            //prepare physical field data
            oldPos.resize(numParticles);
            vel.resize(numParticles);
            lambda.resize(numParticles);
            dpos.resize(numParticles);

            initData();  
        }

        preSolve();
        for (size_t i = 0; i < numSubsteps; i++)
            solve(); 
        postSolve();  


        prim->verts.resize(pos.size());
        for (size_t i = 0; i < pos.size(); i++)
            prim->verts[i] = pos[i]/10.0;//scale to show

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(PBF, {   
                    {
                        {"PrimitiveObject", "prim"},
                        {"float", "dx", "2.51"},
                        {"vec3i", "bound", "40, 40, 40"},
                        // {"int", "numSubsteps", "5"}
                    },
                    {   {"PrimitiveObject", "outPrim"} },
                    {},
                    {"PBD"},
                });
}//namespace zeno