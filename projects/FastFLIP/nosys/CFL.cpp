#include "FLIP_vdb.h"
#include "zeno/VDBGrid.h"
#include <omp.h>
#include <zeno/types/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>
#include <limits>
namespace zeno {

struct CFL : zeno::INode {
  virtual void apply() override {
    auto velocity = get_input("Velocity")->as<VDBFloat3Grid>();
    float dx = get_param<float>("dx");
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

struct Tension_dt : zeno::INode {

  virtual void apply() override {
    float dx = get_param<float>("dx");
    if(has_input("Dx"))
    {
      dx = get_input2<float>("Dx");
    }
    auto density = get_input2<float>("Density");
    auto tension_coef = get_input2<float>("SurfaceTension");

    float dt = std::numeric_limits<float>::max();
    if (tension_coef > 0.f)
    {
      dt = std::sqrt(density*dx*dx*dx / (4.0f*M_PI*tension_coef));
      printf("Surface Tension dt: %f\n", dt);
    }

    auto out_dt = zeno::IObject::make<zeno::NumericObject>();
    out_dt->set<float>(dt);
    set_output("tension_dt", out_dt);
  }
};

static int defCFL =
    zeno::defNodeClass<CFL>("CFL_dt", {/* inputs: */ {
                                           {"VDBGrid", "Velocity"}, {gParamType_Float, "Dx"},
                                       },
                                       /* outputs: */
                                       {
                                           {gParamType_Float, "cfl_dt"},
                                       },
                                       /* params: */
                                       {
                                           {gParamType_Float, "dx", "0.0"},
                                       },

                                       /* category: */
                                       {
                                           "FLIPSolver",
                                       }});



static int defSurfaceTension_dt =
    zeno::defNodeClass<Tension_dt>("SurfaceTension_dt", {
                                    /* inputs: */ 
                                    {
                                      {gParamType_Float, "Dx"},
                                      {gParamType_Float, "Density", "1000.0"},
                                      {gParamType_Float, "SurfaceTension", "0.0"},
                                    },
                                    /* outputs: */
                                    {
                                        {gParamType_Float, "tension_dt"},
                                    },
                                    /* params: */
                                    {
                                        {gParamType_Float, "dx", "0.0"},
                                    },

                                    /* category: */
                                    {
                                        "FLIPSolver",
                                    }});

} // namespace zeno
