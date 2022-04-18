#include "FLIP_vdb.h"
#include <omp.h>
#include <zeno/MeshObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>

/*
static void clamp_liquid_phi_in_solids(openvdb::FloatGrid::Ptr liquid_sdf,
                                                                                  openvdb::FloatGrid::Ptr solid_sdf,
                                                                                  openvdb::FloatGrid::Ptr pushed_out_liquid_sdf,
                                                                                  float dx);
*/
namespace zeno {

struct PushOutLiquidSDF : zeno::INode {
  virtual void apply() override {
    auto dx = std::get<float>(get_param("dx"));
    if(has_input("Dx"))
    {
      dx = get_input("Dx")->as<NumericObject>()->get<float>();
    }
    auto liquid_sdf = get_input("LiquidSDF")->as<VDBFloatGrid>();
    auto solid_sdf = get_input("SolidSDF")->as<VDBFloatGrid>();
    auto pushed_out_liquid_sdf =
        get_input("ExtractedLiquidSDF")->as<VDBFloatGrid>();

    FLIP_vdb::clamp_liquid_phi_in_solids(liquid_sdf->m_grid, solid_sdf->m_grid,
                                         pushed_out_liquid_sdf->m_grid, dx);
  }
};

static int defPushOutLiquidSDF = zeno::defNodeClass<PushOutLiquidSDF>(
    "PushOutLiquidSDF", {/* inputs: */ {"Dx",
                             "LiquidSDF",
                             "SolidSDF",
                             "ExtractedLiquidSDF",
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