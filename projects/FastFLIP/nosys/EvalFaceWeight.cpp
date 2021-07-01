#include "FLIP_vdb.h"
#include <omp.h>
#include <zeno/MeshObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>

/*
void FLIP_vdb::calculate_face_weights(
        openvdb::Vec3fGrid::Ptr face_weight,
        openvdb::FloatGrid::Ptr liquid_sdf,
        openvdb::FloatGrid::Ptr solid_sdf)
*/

namespace zeno {

struct CutCellWeightEval : zeno::INode {
  virtual void apply() override {
    auto face_weight = get_input("FaceWeight")->as<VDBFloat3Grid>();
    auto liquid_sdf = get_input("LiquidSDF")->as<VDBFloatGrid>();
    auto solid_sdf = get_input("SolidSDF")->as<VDBFloatGrid>();
    FLIP_vdb::calculate_face_weights(face_weight->m_grid, liquid_sdf->m_grid,
                                     solid_sdf->m_grid);
  }
};

static int defCutCellWeightEval =
    zeno::defNodeClass<CutCellWeightEval>("CutCellWeight", {/* inputs: */ {
                                                                "LiquidSDF",
                                                                "SolidSDF",
                                                                "FaceWeight",
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