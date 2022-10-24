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

struct NonNewtonViscosity : zeno::INode {
  void apply() override {
    auto dx = get_param<float>("dx");
    if(has_input("Dx"))
    {
      dx = get_input2<float>("Dx");
    }

    float mu_0 = get_input2<float>("Viscosity");
    float mu_inf = get_input2<float>("Viscosity_inf");
    float scale = get_input2<float>("scale");
    float alpha = get_input2<float>("smoothness");
    float n = get_input2<float>("n");
    auto velocity = get_input<VDBFloat3Grid>("Velocity");
    auto viscosity = get_input<VDBFloatGrid>("ViscosityGrid");

    FLIP_vdb::make_non_newton_viscosity_field(
              viscosity->m_grid, velocity->m_grid,
              mu_0, mu_inf, scale, alpha, n, dx);
  }
};

ZENDEFNODE(NonNewtonViscosity, {
                                /* inputs: */
                                {"Dx",
                                 {"float", "Viscosity", "0.0"},
                                 {"float", "Viscosity_inf", "0.0"},
                                 {"float", "scale", "1.0"},
                                 {"float", "smoothness", "1.0"},
                                 {"float", "thickening", "1.0"},
                                 "Velocity",
                                 "ViscosityGrid"},
                                /* outputs: */
                                {},
                                /* params: */
                                {{"float", "dx", "0.0"}},
                                /* category: */
                                {"FLIPSolver"},
                               });

} // namespace zeno