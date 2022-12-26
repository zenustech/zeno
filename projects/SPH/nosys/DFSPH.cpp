#include <iostream>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include "SPlisHSPlasH/SPlisHSPlasH/DFSPH/TimeStepDFSPH.h"
namespace zeno
{
struct DFSPH : INode
{
    virtual void apply() override
    {
        std::cout<<"DFSPH!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
        std::cout<<"DFSPH!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
        std::cout<<"DFSPH!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
        std::cout<<"DFSPH!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
    }
};
    


ZENDEFNODE(DFSPH, {   
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
                    },
                    {   {"PrimitiveObject", "outPrim"} },
                    {},
                    {"SPH"},
                });

}//namespace zeno