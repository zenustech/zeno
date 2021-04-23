#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <zen/NumericObject.h>
#include <zen/VDBGrid.h>
#include <omp.h>
#include "FLIP_vdb.h"


/*void FLIP_vdb::field_add_vector(openvdb::FloatGrid::Ptr velocity_field,
	openvdb::FloatGrid::Ptr face_weight,
	float x, float y, float z, float dt)
*/

namespace zenbase{
    
    struct FieldAddVector : zen::INode{
        virtual void apply() override {
            auto dt = get_input("dt")->as<zenbase::NumericObject>()->get<float>();
            float vx = std::get<float>(get_param("vx"));
            float vy = std::get<float>(get_param("vy"));
            float vz = std::get<float>(get_param("vz"));
            auto velocity = get_input("Velocity")->as<VDBFloat3Grid>();
            if(has_input("FieldWeight")) {
                auto face_weight = get_input("FieldWeight")->as<VDBFloat3Grid>();
            
                FLIP_vdb::field_add_vector(velocity->m_grid, 
                face_weight->m_grid, vx, vy, vz, dt);
            }
            else {
                using TmpT = decltype(std::declval<VDBFloat3Grid>().m_grid);
                TmpT tmp{nullptr};
                FLIP_vdb::field_add_vector(velocity->m_grid, 
                tmp, vx, vy, vz, dt);
            }
        }
    };

static int defFieldAddVector = zen::defNodeClass<FieldAddVector>("FieldAddVector",
    { /* inputs: */ {
        "dt", "Velocity", "FieldWeight", 
    }, 
    /* outputs: */ {
    }, 
    /* params: */ {
       {"float", "vx", "0.0"},
       {"float", "vy", "0.0"},
       {"float", "vz", "0.0"},
    }, 
    
    /* category: */ {
    "FLIPSolver",
    }});

}