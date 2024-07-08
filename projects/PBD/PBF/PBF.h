#pragma once

#include <zeno/zeno.h>
#include <map>
#include <zeno/types/PrimitiveObject.h>
#include "SPHKernelFuncs.h"

namespace zeno{
struct PBF : INode{
//params
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


    //physical fields
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

    //data for cells
    inline vec3i getCellXYZ(const vec3f& p);
    inline int getCellID(const vec3f& p);
    inline int getCellHash(int i, int j, int k);
    inline bool isInBound(const vec3i& cell);
    inline int cellXYZ2ID(const vec3i& xyz);
    inline vec3i cellID2XYZ(int i);
    std::array<int, 3> numCellXYZ;
    int numCell;
    float dx; //cell size
    float dxInv; 
    void initCellData();
    void initNeighborList();
    struct Cell
    {
        int x,y,z;
        std::vector<int> parInCell; 
    };
    std::map<int, Cell>  cell;

    //neighborList
    std::vector<std::vector<int>> neighborList;
    void neighborSearch();

public:
    void setParams()
    {
        //用户自定义参数
        dt = get_input<zeno::NumericObject>("dt")->get<float>();
        pRadius = get_input<zeno::NumericObject>("particle_radius")->get<float>();
        bounds_min = get_input<zeno::NumericObject>("bounds_min")->get<zeno::vec3f>();
        bounds_max = get_input<zeno::NumericObject>("bounds_max")->get<zeno::vec3f>();
        gravity = get_input<zeno::NumericObject>("gravity")->get<zeno::vec3f>();
        rho0 = get_input<zeno::NumericObject>("rho0")->get<float>();
        lambdaEpsilon = get_input<zeno::NumericObject>("lambdaEpsilon")->get<float>();
        coeffDq = get_input<zeno::NumericObject>("coeffDq")->get<float>();
        coeffK = get_input<zeno::NumericObject>("coeffK")->get<float>();

        dx = get_input<zeno::NumericObject>("dx")->get<float>();
        
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

            initCellData();
            initNeighborList(); 
        }

        preSolve();
        neighborSearch();//grid-baed neighborSearch
        for (size_t i = 0; i < numSubsteps; i++)
            solve(); 
        postSolve();  

        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(PBF, {   
                    {
                        {"PrimitiveObject", "prim"},
                        {"float", "dx", "2.51"},
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