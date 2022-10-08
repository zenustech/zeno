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

        velocity->m_packedGrid->from_vec3(velocity->m_grid);

        vdb_velocity_extrapolator::union_extrapolate(n,
		                            velocity->m_packedGrid->v[0],
		                            velocity->m_packedGrid->v[1],
		                            velocity->m_packedGrid->v[2],
	  	                            &(liquidsdf->m_grid->tree()));

        velocity->m_packedGrid->to_vec3(velocity->m_grid);
        
        //vdb_velocity_extrapolator::extrapolate(n, velocity->m_grid);
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