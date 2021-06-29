#include <zeno/zen.h>
#include <zeno/VDBGrid.h>
#include <omp.h>
#include "FLIP_vdb.h"

namespace zen{
    
    struct FluidReseed : zen::INode{
        virtual void apply() override {
            auto particles = get_input("Particles")->as<VDBPointsGrid>();
            auto liquidSDF = get_input("LiquidSDF")->as<VDBFloatGrid>();
            auto liquidVel = get_input("FluidVel")->as<VDBFloat3Grid>();
            FLIP_vdb::reseed_fluid (particles->m_grid, liquidSDF->m_grid, liquidVel->m_grid);
        }
    };

static int defFluidReseed = zen::defNodeClass<FluidReseed>("FluidReseed",
    { /* inputs: */ {
        "Particles", "LiquidSDF", "FluidVel",
    }, 
    /* outputs: */ {
    }, 
    /* params: */ {
    }, 
    
    /* category: */ {
    "FLIPSolver",
    }});

}