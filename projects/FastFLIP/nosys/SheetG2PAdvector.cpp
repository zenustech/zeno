#include <zeno/zeno.h>
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/VDBGrid.h>
#include <omp.h>
#include "FLIP_vdb.h"

//static void Advect(float dt, openvdb::points::PointDataGrid::Ptr m_particles, openvdb::Vec3fGrid::Ptr m_velocity,
//				   openvdb::Vec3fGrid::Ptr m_velocity_after_p2g, float pic_component, int RK_ORDER);
namespace zen{
    
    struct G2PAdvectorSheet : zeno::INode{
        virtual void apply() override {
            auto dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
            auto dx = std::get<float>(get_param("dx"));
            auto surfaceSize = std::get<int>(get_param("surface_size"));
            auto smoothness = std::get<float>(get_param("pic_smoothness"));
            auto RK_ORDER = std::get<int>(get_param("RK_ORDER"));
            
            auto particles = get_input("Particles")->as<VDBPointsGrid>();
            auto velocity = get_input("Velocity")->as<VDBFloat3Grid>();
            auto liquidsdf = get_input("LiquidSDF")->as<VDBFloatGrid>();
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
            FLIP_vdb::AdvectSheetty(dt, dx, (float)surfaceSize * dx, particles->m_grid, liquidsdf->m_grid, velocity->m_grid, velocity_after_p2g->m_grid, solid_sdf->m_grid, solid_vel->m_grid, smoothness, RK_ORDER);
        }
    };

static int defG2PAdvectorSheet = zeno::defNodeClass<G2PAdvectorSheet>("G2PAdvectorSheetty",
    { /* inputs: */ {
        "dt","Particles", "Velocity", "LiquidSDF", "PostAdvVelocity", "SolidSDF", "SolidVelocity",
    }, 
    /* outputs: */ {
    }, 
    /* params: */ {
      {"float", "dx", "0.01 0.0"},
      {"int", "RK_ORDER", "1 1 4"},
      {"float", "pic_smoothness", "0.1 0.0 1.0"},
      {"int", "surface_size", "4 0 8"},
    }, 
    
    /* category: */ {
    "FLIPSolver",
    }});

}
