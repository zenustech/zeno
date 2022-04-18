#include "FLIP_vdb.h"
#include "zeno/VDBGrid.h"
#include <omp.h>
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>
namespace zeno {

struct CFL : zeno::INode {
  virtual void apply() override {
    auto velocity = get_input("Velocity")->as<VDBFloat3Grid>();
    float dx = get_param<float>(("dx"));
    if(has_input("Dx"))
    {
      dx = get_input("Dx")->as<NumericObject>()->get<float>();
    }
    float dt = FLIP_vdb::cfl(velocity->m_grid);
    printf("CFL dt: %f\n", dt);
    auto out_dt = zeno::IObject::make<zeno::NumericObject>();
    float scaling = dx / velocity->m_grid->voxelSize()[0];
    out_dt->set<float>(scaling * dt);
    set_output("cfl_dt", out_dt);
  }
};

static int defCFL =
    zeno::defNodeClass<CFL>("CFL_dt", {/* inputs: */ {
                                           "Velocity", "Dx",
                                       },
                                       /* outputs: */
                                       {
                                           "cfl_dt",
                                       },
                                       /* params: */
                                       {
                                           {"float", "dx", "0.0"},
                                       },

                                       /* category: */
                                       {
                                           "FLIPSolver",
                                       }});

} // namespace zeno
