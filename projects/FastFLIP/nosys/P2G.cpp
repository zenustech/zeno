#include "FLIP_vdb.h"
#include "../vdb_velocity_extrapolator.h"
#include <omp.h>
#include <zeno/types/MeshObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>

namespace zeno {

struct FLIP_P2G : zeno::INode {
  virtual void apply() override {
    auto dx = get_param<float>("dx");
    auto n = get_param<int>("VelExtraLayer");

    if(has_input("Dx"))
    {
      dx = get_input("Dx")->as<NumericObject>()->get<float>();
    }
    auto Particles = get_input("Particles")->as<VDBPointsGrid>();
    auto VelGrid = get_input("Velocity")->as<VDBFloat3Grid>();
    auto PostP2GVelGrid = get_input("PostP2GVelocity")->as<VDBFloat3Grid>();
    auto LiquidSDFGrid = get_input("LiquidSDF")->as<VDBFloatGrid>();

    packed_FloatGrid3 packed_VelGrid, packed_PostP2GVelGrid;
    packed_VelGrid.from_vec3(VelGrid->m_grid);
    packed_PostP2GVelGrid.from_vec3(PostP2GVelGrid->m_grid);

    FLIP_vdb::particle_to_grid_collect_style(
        packed_VelGrid, packed_PostP2GVelGrid,
        LiquidSDFGrid->m_grid, Particles->m_grid, dx);

    vdb_velocity_extrapolator::union_extrapolate(n,
		                            packed_VelGrid.v[0],
		                            packed_VelGrid.v[1],
		                            packed_VelGrid.v[2],
	  	                          &(LiquidSDFGrid->m_grid->tree()));

    packed_VelGrid.to_vec3(VelGrid->m_grid);
    packed_PostP2GVelGrid.to_vec3(PostP2GVelGrid->m_grid);
  }
};

static int defFLIP_P2G =
    zeno::defNodeClass<FLIP_P2G>("FLIP_P2G", {/* inputs: */ {
                                                  {gParamType_Float, "Dx"},
                                                  {gParamType_VDBGrid, "Particles"},
                                                  {gParamType_VDBGrid, "Velocity"},
                                                  {gParamType_VDBGrid, "PostP2GVelocity"},
                                                  {gParamType_VDBGrid, "LiquidSDF"},
                                              },
                                              /* outputs: */ {},
                                              /* params: */
                                              {
                                                  {gParamType_Float, "dx", "0.01 0.0"},
                                                  {gParamType_Int, "VelExtraLayer", "3"},
                                              },

                                              /* category: */
                                              {
                                                  "FLIPSolver",
                                              }});

} // namespace zeno