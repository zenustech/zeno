#include <zen/zen.h>
#include <zen/VDBGrid.h>
#include <omp.h>
#include "FLIP_vdb.h"

namespace zenbase{
    
    struct ParticleEmitter : zen::INode{
        virtual void apply() override {
            auto particles = get_input("Particles")->as<VDBPointsGrid>();
            auto shape = get_input("ShapeSDF")->as<VDBFloatGrid>();
            //auto velocityVolume = get_input("VelocityVolume")->as<VDBFloat3Grid>();
            float vx = std::get<float>(get_param("vx"));
            float vy = std::get<float>(get_param("vy"));
            float vz = std::get<float>(get_param("vz"));
            // if(velocityVolume==nullptr)
            // {
            FLIP_vdb::emit_liquid(particles->m_grid, shape->m_grid, nullptr, vx, vy,vz);
            // } else {
            //     FLIP_vdb::emit_liquid(particles->m_grid, shape->m_grid, velocityVolume->m_grid, 0, 0,0);
            // }
        }
    };

static int defParticleEmitter = zen::defNodeClass<ParticleEmitter>("ParticleEmitter",
    { /* inputs: */ {
        "Particles", "ShapeSDF", //"VelocityVolume", 
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