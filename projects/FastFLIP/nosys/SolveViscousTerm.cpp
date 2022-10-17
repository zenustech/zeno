#include "FLIP_vdb.h"
#include "../vdb_velocity_extrapolator.h"
#include <omp.h>
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>

/*
static void solve_viscosity(
    packed_FloatGrid3 &velocity,
    packed_FloatGrid3 &viscous_velocity,
    openvdb::FloatGrid::Ptr &liquid_sdf,
    openvdb::FloatGrid::Ptr &solid_sdf,
    openvdb::Vec3fGrid::Ptr &solid_velocity,
    float density, float viscosity, float dt);
*/

namespace zeno {

struct SolveViscousTerm : zeno::INode {
  void apply() override {
    auto n = get_param<int>("VelExtraLayer");
    auto dt = get_input2<float>("dt");
    auto dx = get_param<float>("dx");
    if(has_input("Dx"))
    {
      dx = get_input2<float>("Dx");
    }
    auto density = get_input2<float>("Density");
    auto viscosity = get_input2<float>("Viscosity");
    auto velocity = get_input<VDBFloat3Grid>("Velocity");
    auto velocity_viscous = get_input<VDBFloat3Grid>("ViscousVelocity");
    auto liquid_sdf = get_input<VDBFloatGrid>("LiquidSDF");
    auto solid_sdf = get_input<VDBFloatGrid>("SolidSDF");
    auto solid_velocity = get_input<VDBFloat3Grid>("SolidVelocity");

    if (viscosity > 0)
    {
      packed_FloatGrid3 packed_velocity, packed_viscous_vel;
      packed_velocity.from_vec3(velocity->m_grid);
      packed_viscous_vel.from_vec3(velocity_viscous->m_grid);

      FLIP_vdb::solve_viscosity(packed_velocity, packed_viscous_vel, liquid_sdf->m_grid,
                                solid_sdf->m_grid, solid_velocity->m_grid,
                                density, viscosity, dt);

      vdb_velocity_extrapolator::union_extrapolate(n,
  		                            packed_viscous_vel.v[0],
  		                            packed_viscous_vel.v[1],
  		                            packed_viscous_vel.v[2],
  	  	                          &(liquid_sdf->m_grid->tree()));

      packed_viscous_vel.to_vec3(velocity_viscous->m_grid);
    }
    else
    {
      velocity_viscous->m_grid = velocity->m_grid->deepCopy();
      velocity_viscous->setName("Velocity_Viscous");
    }
  }
};


ZENDEFNODE(SolveViscousTerm, {
                                /* inputs: */
                                {"dt", "Dx",
                                 {"float", "Density", "1000.0"},
                                 {"float", "Viscosity", "0.0"},
                                 "Velocity",
                                 "ViscousVelocity",
                                 "LiquidSDF",
                                 "SolidSDF",
                                 "SolidVelocity"},
                                /* outputs: */
                                {},
                                /* params: */
                                {{"float", "dx", "0.0"},
                                 {"int", "VelExtraLayer", "3"}},
                                /* category: */
                                {"FLIPSolver"},
                             });

} // namespace zeno
