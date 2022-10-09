#include "FLIP_vdb.h"
#include "../vdb_velocity_extrapolator.h"
#include <omp.h>
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>

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

namespace zeno {

struct SubtractPressureGradient : zeno::INode {
  virtual void apply() override {

    auto dx = get_param<float>("dx");
    if(has_input("Dx"))
    {
      dx = get_input("Dx")->as<NumericObject>()->get<float>();
    }
    auto dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
    auto liquid_sdf = get_input("LiquidSDF")->as<VDBFloatGrid>();
    auto solid_sdf = get_input("SolidSDF")->as<VDBFloatGrid>();
    auto curr_pressure = get_input("Pressure")->as<VDBFloatGrid>();
    auto face_weight = get_input("CellFWeight")->as<VDBFloat3Grid>();
    auto velocity = get_input("Velocity")->as<VDBFloat3Grid>();
    auto solid_velocity = get_input("SolidVelocity")->as<VDBFloat3Grid>();

    packed_FloatGrid3 packed_velocity;
    packed_velocity.from_vec3(velocity->m_grid);

    FLIP_vdb::apply_pressure_gradient(
        liquid_sdf->m_grid, solid_sdf->m_grid,
        curr_pressure->m_grid, face_weight->m_grid,
        packed_velocity,
        solid_velocity->m_grid, dx, dt);

    vdb_velocity_extrapolator::union_extrapolate(3,
		                            packed_velocity.v[0],
		                            packed_velocity.v[1],
		                            packed_velocity.v[2],
	  	                          &(liquid_sdf->m_grid->tree()));

    packed_velocity.to_vec3(velocity->m_grid);
  }
};

static int defSubtractPressureGradient =
    zeno::defNodeClass<SubtractPressureGradient>("SubtractPressureGradient",
                                                 {/* inputs: */ {
                                                      "dt","Dx",
                                                      "LiquidSDF",
                                                      "SolidSDF",
                                                      "Pressure",
                                                      "CellFWeight",
                                                      "Velocity",
                                                      "SolidVelocity",

                                                  },
                                                  /* outputs: */ {},
                                                  /* params: */
                                                  {
                                                      {"float", "dx", "0.0"},
                                                  },

                                                  /* category: */
                                                  {
                                                      "FLIPSolver",
                                                  }});

} // namespace zeno
