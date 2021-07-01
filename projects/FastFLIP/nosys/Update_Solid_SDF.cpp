#include "FLIP_vdb.h"
#include <omp.h>
#include <zeno/MeshObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
// void FLIP_vdb::update_solid_sdf(std::vector<openvdb::FloatGrid::Ptr>
// &moving_solids, openvdb::points::PointDataGrid::Ptr particles)

namespace zeno {

struct FLIP_Solid_Modifier : zeno::INode {
  virtual void apply() override {
    auto particles = get_input("Particles")->as<VDBPointsGrid>();

    VDBFloatGrid *moving_solid_sdf = nullptr;
    if (has_input("DynaSolid_SDF"))
      moving_solid_sdf = get_input("DynaSolid_SDF")->as<VDBFloatGrid>();

    VDBFloatGrid *static_solid_sdf;
    static_solid_sdf = get_input("Static_SDF")->as<VDBFloatGrid>();

    std::vector<openvdb::FloatGrid::Ptr> moving_solids;
    if (moving_solid_sdf != nullptr)
      moving_solids.emplace_back(moving_solid_sdf->m_grid);

    FLIP_vdb::update_solid_sdf(moving_solids, static_solid_sdf->m_grid,
                               particles->m_grid);
  }
};

static int defFLIP_Solid_Modifier = zeno::defNodeClass<FLIP_Solid_Modifier>(
    "FLIPApplyBoundary", {/* inputs: */ {
                              "Particles",
                              "DynaSolid_SDF",
                              "StatSolid_SDF",
                          },
                          /* outputs: */ {},
                          /* params: */
                          {

                          },

                          /* category: */
                          {
                              "FLIPSolver",
                          }});

} // namespace zeno