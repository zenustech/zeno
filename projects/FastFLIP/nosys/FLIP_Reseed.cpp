#include "FLIP_vdb.h"
#include <omp.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>

namespace zeno {

struct FluidReseed : zeno::INode {
  virtual void apply() override {
    auto particles = get_input("Particles")->as<VDBPointsGrid>();
    auto liquidSDF = get_input("LiquidSDF")->as<VDBFloatGrid>();
    auto liquidVel = get_input("FluidVel")->as<VDBFloat3Grid>();
    FLIP_vdb::reseed_fluid(particles->m_grid, liquidSDF->m_grid,
                           liquidVel->m_grid);
  }
};

static int defFluidReseed =
    zeno::defNodeClass<FluidReseed>("FluidReseed", {/* inputs: */ {
                                                        "Particles",
                                                        "LiquidSDF",
                                                        "FluidVel",
                                                    },
                                                    /* outputs: */ {},
                                                    /* params: */ {},

                                                    /* category: */
                                                    {
                                                        "FLIPSolver",
                                                    }});

} // namespace zeno