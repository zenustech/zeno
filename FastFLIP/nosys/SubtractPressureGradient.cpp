#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <zen/VDBGrid.h>
#include <omp.h>
#include "FLIP_vdb.h"
/*
static void apply_pressure_gradient(
	openvdb::FloatGrid::Ptr liquid_sdf,
	openvdb::FloatGrid::Ptr solid_sdf,
	openvdb::FloatGrid::Ptr pushed_out_liquid_sdf,
	openvdb::FloatGrid::Ptr pressure,
	openvdb::Vec3fGrid::Ptr face_weight,
	openvdb::Vec3fGrid::Ptr velocity,
	openvdb::Vec3fGrid::Ptr solid_velocity,
	float dx,float dt);
*/

namespace zenbase{
    
    struct SubtractPressureGradient : zen::INode{
        virtual void apply() override {
            auto dt = std::get<float>(get_param("dt"));
            auto dx = std::get<float>(get_param("dx"));
            auto liquid_sdf            = get_input("LiquidSDF"         )->as<VDBFloatGrid>();
            auto solid_sdf             = get_input("SolidSDF"          )->as<VDBFloatGrid>();
            auto pushed_out_liquid_sdf = get_input("ExtractedLiquidSDF")->as<VDBFloatGrid>();
            auto curr_pressure         = get_input("Pressure"          )->as<VDBFloatGrid>();
            auto face_weight           = get_input("CellFWeight"       )->as<VDBFloat3Grid>();
            auto velocity              = get_input("Velocity"          )->as<VDBFloat3Grid>(); 
            auto solid_velocity        = get_input("SolidVelocity"     )->as<VDBFloat3Grid>();

            FLIP_vdb::apply_pressure_gradient(
                liquid_sdf            -> m_grid,
                solid_sdf             -> m_grid,
                pushed_out_liquid_sdf -> m_grid,
                curr_pressure         -> m_grid,
                face_weight           -> m_grid,
                velocity              -> m_grid,
                solid_velocity        -> m_grid,
                dx,dt
            );
        }
    };

static int defSubtractPressureGradient = zen::defNodeClass<SubtractPressureGradient>("SubtractPressureGradient",
    { /* inputs: */ {
        "LiquidSDF"         ,
        "SolidSDF"          ,
        "ExtractedLiquidSDF",
        "Pressure"          ,
        "CellFWeight"       ,
        "Velocity"          ,
        "SolidVelocity"     ,

    }, 
    /* outputs: */ {
    }, 
    /* params: */ {
       {"float", "dt", "0.0"},
       {"float", "dx", "0.0"},
    }, 
    
    /* category: */ {
    "FLIPSolver",
    }});

}