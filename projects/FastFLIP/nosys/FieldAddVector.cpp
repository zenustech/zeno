#include "FLIP_vdb.h"
#include <omp.h>
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>

/*void FLIP_vdb::field_add_vector(openvdb::FloatGrid::Ptr velocity_field,
        openvdb::FloatGrid::Ptr face_weight,
        float x, float y, float z, float dt)
*/

namespace zeno {

struct FieldAddVector : zeno::INode {
  virtual void apply() override {
    // auto dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
    // float vx = get_param<float>("vx");
    // float vy = get_param<float>("vy");
    // float vz = get_param<float>("vz");
    auto ivec3 =
        get_input("invec3")->as<zeno::NumericObject>()->get<zeno::vec3f>();
    auto velocity = get_input("Velocity")->as<VDBFloat3Grid>();

    packed_FloatGrid3 packed_velocity;
    packed_velocity.from_vec3(velocity->m_grid);
    
    FLIP_vdb::field_add_vector( packed_velocity,
                                ivec3[0], ivec3[1], ivec3[2], 1.0);
    
    packed_velocity.to_vec3(velocity->m_grid);
  }
};

static int defFieldAddVector =
    zeno::defNodeClass<FieldAddVector>("FieldAddVector", {/* inputs: */ {
                                                              "invec3",
                                                              "Velocity",
                                                              "FieldWeight",
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
