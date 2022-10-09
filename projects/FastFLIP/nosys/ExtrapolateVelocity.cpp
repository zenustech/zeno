#include "openvdb/openvdb.h"
#include <omp.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/tools/Morphology.h>
#include <zeno/VDBGrid.h>
#include <zeno/ParticlesObject.h>
#include <zeno/zeno.h>

#include "../vdb_velocity_extrapolator.h"

namespace zeno {

struct Vec3FieldExtrapolate : zeno::INode {
    virtual void apply() override {
        int n = get_param<int>("NumIterates");
        auto velocity = get_input("Field")->as<VDBFloat3Grid>();
        auto liquidsdf = get_input("LiquidSDF")->as<VDBFloatGrid>();
#if 0
        vdb_velocity_extrapolator::union_extrapolate(n,
		                            velocity->v[0],
		                            velocity->v[1],
		                            velocity->v[2],
	  	                            &(liquidsdf->m_grid->tree()));
#endif
        
        vdb_velocity_extrapolator::extrapolate(n, velocity->m_grid);
    }
};

static int defVec3FieldExtrapolate =
    zeno::defNodeClass<Vec3FieldExtrapolate>("Vec3FieldExtrapolate", {/* inputs: */ {
                                                                          "Field",
                                                                          "LiquidSDF",
                                                                      },
                                                                      /* outputs: */ {},
                                                                      /* params: */
                                                                      {
                                                                          {"int", "NumIterates", "1"},
                                                                      },

                                                                      /* category: */
                                                                      {
                                                                          "FLIPSolver",
                                                                      }});

} // namespace zeno