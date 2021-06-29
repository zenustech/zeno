#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <omp.h>
#include "FLIP_vdb.h"

namespace zen{
    
    struct ParticleEmitter : zen::INode{
        virtual void apply() override {
            auto particles = get_input("Particles")->as<VDBPointsGrid>();
            auto shape = get_input("ShapeSDF")->as<VDBFloatGrid>();
            float vx = std::get<float>(get_param("vx"));
            float vy = std::get<float>(get_param("vy"));
            float vz = std::get<float>(get_param("vz"));

            VDBFloat3Grid* velocityVolume = new VDBFloat3Grid();
            if(has_input("VelocityVolume")){
                velocityVolume = get_input("VelocityVolume")->as<VDBFloat3Grid>();
                
            } else {
                velocityVolume->m_grid = nullptr;
            }

            VDBFloatGrid* liquid_sdf = new VDBFloatGrid();;
            if(has_input("LiquidSDF")){
                liquid_sdf = get_input("LiquidSDF")->as<VDBFloatGrid>();
                
            } else {
                liquid_sdf->m_grid = nullptr;
            }
            FLIP_vdb::emit_liquid(particles->m_grid, shape->m_grid, velocityVolume->m_grid, liquid_sdf->m_grid, vx, vy, vz);
        }
    };

static int defParticleEmitter = zen::defNodeClass<ParticleEmitter>("ParticleEmitter",
    { /* inputs: */ {
        "Particles", "ShapeSDF", "VelocityVolume", "LiquidSDF",
    }, 
    /* outputs: */ {
    }, 
    /* params: */ {
       {"float", "vx", "0.0"},
       {"float", "vy", "0.0"},
       {"float", "vz", "0.0"},
    }, 
    
    /* category: */ {
    "FLIPSolver",
    }});

}