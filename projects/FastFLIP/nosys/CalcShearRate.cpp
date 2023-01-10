#include "FLIP_vdb.h"
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>

/*
static auto make_non_newton_viscosity_field(
    openvdb::Vec3fGrid::Ptr &velocity,
    float mu_0, float mu_inf, float scale, float alpha, float n,
    float dx);
*/

namespace zeno {

struct CalcShearRate : zeno::INode {
  void apply() override {
    auto velocity = get_input<VDBFloat3Grid>("Velocity");
    auto shear_rate = get_input<VDBFloatGrid>("ShearRate");
    auto liquid_sdf = get_input<VDBFloatGrid>("LiquidSDF");

    FLIP_vdb::calculate_shear_rate(shear_rate->m_grid, velocity->m_grid, liquid_sdf->m_grid);
  }
};

ZENDEFNODE(CalcShearRate, {
                            /* inputs: */
                            {"Velocity",
                             "ShearRate",
                             "LiquidSDF"},
                            /* outputs: */
                            {},
                            /* params: */
                            {},
                            /* category: */
                            {"FLIPSolver"},
                          });

} // namespace zeno