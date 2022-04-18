#include "FLIP_vdb.h"
#include <omp.h>
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>

/*
static void FLIP_vdb::solve_pressure_simd(
        openvdb::FloatGrid::Ptr liquid_sdf,
        openvdb::FloatGrid::Ptr pushed_out_liquid_sdf,
        openvdb::FloatGrid::Ptr rhsgrid,
        openvdb::FloatGrid::Ptr curr_pressure,
        openvdb::Vec3fGrid::Ptr face_weight,
        openvdb::Vec3fGrid::Ptr velocity,
        openvdb::Vec3fGrid::Ptr solid_velocity,
        float dt, float dx);
*/

namespace zeno {

struct AssembleSolvePPE : zeno::INode {
  virtual void apply() override {
    auto dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
    auto dx = get_param<float>(("dx"));
    if(has_input("Dx"))
    {
      dx = get_input("Dx")->as<NumericObject>()->get<float>();
    }
    auto liquid_sdf = get_input("LiquidSDF")->as<VDBFloatGrid>();
    auto pushed_out_liquid_sdf =
        get_input("ExtractedLiquidSDF")->as<VDBFloatGrid>();
    auto rhsgrid = get_input("Divergence")->as<VDBFloatGrid>();
    auto curr_pressure = get_input("Pressure")->as<VDBFloatGrid>();
    auto face_weight = get_input("CellFWeight")->as<VDBFloat3Grid>();
    auto velocity = get_input("Velocity")->as<VDBFloat3Grid>();
    auto solid_velocity = get_input("SolidVelocity")->as<VDBFloat3Grid>();
    FLIP_vdb::solve_pressure_simd(
        liquid_sdf->m_grid, pushed_out_liquid_sdf->m_grid, rhsgrid->m_grid,
        curr_pressure->m_grid, face_weight->m_grid, velocity->m_grid,
        solid_velocity->m_grid, dt, dx);
  }
};

static int defAssembleSolvePPE = zeno::defNodeClass<AssembleSolvePPE>(
    "AssembleSolvePPE", {/* inputs: */ {
                             "dt","Dx",
                             "LiquidSDF",
                             "ExtractedLiquidSDF",
                             "Divergence",
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
