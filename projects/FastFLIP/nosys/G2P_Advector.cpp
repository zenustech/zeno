#include <zeno/zen.h>
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/VDBGrid.h>
#include <omp.h>
#include "FLIP_vdb.h"
// I finally decide put this P2G_Advector into FastFLIP, to reduce modularizing effort..
//static void Advect(float dt, openvdb::points::PointDataGrid::Ptr m_particles, openvdb::Vec3fGrid::Ptr m_velocity,
//				   openvdb::Vec3fGrid::Ptr m_velocity_after_p2g, float pic_component, int RK_ORDER);
namespace zen{
    
    struct G2P_Advector : zen::INode{
        virtual void apply() override {
            auto dt = get_input("dt")->as<zen::NumericObject>()->get<float>();
            auto dx = std::get<float>(get_param("dx"));
            auto smoothness = std::get<float>(get_param("pic_smoothness"));
            auto RK_ORDER = std::get<int>(get_param("RK_ORDER"));
            
            auto particles = get_input("Particles")->as<VDBPointsGrid>();
            auto velocity = get_input("Velocity")->as<VDBFloat3Grid>();
            VDBFloatGrid* solid_sdf = new VDBFloatGrid();
            if(has_input("SolidSDF"))
                solid_sdf = get_input("SolidSDF")->as<VDBFloatGrid>();
            else
                solid_sdf->m_grid = nullptr;
            VDBFloat3Grid* solid_vel = new VDBFloat3Grid();
            if(has_input("SolidVelocity"))
                solid_vel = get_input("SolidVelocity")->as<VDBFloat3Grid>();
            else
                solid_vel->m_grid = nullptr;
            auto velocity_after_p2g = get_input("PostAdvVelocity")->as<VDBFloat3Grid>();
            FLIP_vdb::Advect(dt, dx, particles->m_grid, velocity->m_grid, velocity_after_p2g->m_grid, solid_sdf->m_grid, solid_vel->m_grid, smoothness, RK_ORDER);
        }
    };

static int defG2P_Advector = zen::defNodeClass<G2P_Advector>("G2P_Advector",
    { /* inputs: */ {
        "dt","Particles", "Velocity", "PostAdvVelocity", "SolidSDF", "SolidVelocity",
    }, 
    /* outputs: */ {
    }, 
    /* params: */ {
      {"float", "dx", "0.01 0.0"},
      {"int", "RK_ORDER", "1 1 4"},
      {"float", "pic_smoothness", "0.02 0.0 1.0"},
    }, 
    
    /* category: */ {
    "FLIPSolver",
    }});

}
