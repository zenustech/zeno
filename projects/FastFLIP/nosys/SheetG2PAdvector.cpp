#include "FLIP_vdb.h"
#include <omp.h>
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>


// static void Advect(float dt, openvdb::points::PointDataGrid::Ptr m_particles,
// openvdb::Vec3fGrid::Ptr m_velocity, 				   openvdb::Vec3fGrid::Ptr
//m_velocity_after_p2g, float pic_component, int RK_ORDER);
namespace zeno {

struct G2PAdvectorSheet : zeno::INode {
  virtual void apply() override {
    auto dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
    auto dx = get_param<float>(("dx"));
    if(has_input("Dx"))
    {
      dx = get_input("Dx")->as<NumericObject>()->get<float>();
    }
    auto surfaceSize = get_param<int>(("surface_size"));
    auto smoothness = get_param<float>(("pic_smoothness"));
    auto RK_ORDER = get_param<int>(("RK_ORDER"));

    auto particles = get_input("Particles")->as<VDBPointsGrid>();
    auto velocity = get_input("Velocity")->as<VDBFloat3Grid>();
    auto liquidsdf = get_input("LiquidSDF")->as<VDBFloatGrid>();
    openvdb::FloatGrid::Ptr solid_sdf;
    if (has_input("SolidSDF"))
      solid_sdf = get_input("SolidSDF")->as<VDBFloatGrid>()->m_grid;
    else
      solid_sdf = nullptr;

    openvdb::Vec3fGrid::Ptr solid_vel;
    if (has_input("SolidVelocity"))
      solid_vel = get_input("SolidVelocity")->as<VDBFloat3Grid>()->m_grid;
    else
      solid_vel = nullptr;
    auto velocity_after_p2g = get_input("PostAdvVelocity")->as<VDBFloat3Grid>();
    FLIP_vdb::AdvectSheetty(dt, dx, (float)surfaceSize * dx, particles->m_grid,
                            liquidsdf->m_grid, velocity->m_grid,
                            velocity_after_p2g->m_grid, solid_sdf,
                            solid_vel, smoothness, RK_ORDER);
  }
};

static int defG2PAdvectorSheet = zeno::defNodeClass<G2PAdvectorSheet>(
    "G2PAdvectorSheetty", {/* inputs: */ {
                               "dt","Dx",
                               "Particles",
                               "Velocity",
                               "LiquidSDF",
                               "PostAdvVelocity",
                               "SolidSDF",
                               "SolidVelocity",
                           },
                           /* outputs: */ {},
                           /* params: */
                           {
                               {"float", "dx", "0.01 0.0"},
                               {"int", "RK_ORDER", "1 1 4"},
                               {"float", "pic_smoothness", "0.1 0.0 1.0"},
                               {"int", "surface_size", "4 0 8"},
                           },

                           /* category: */
                           {
                               "FLIPSolver",
                           }});

} // namespace zeno
