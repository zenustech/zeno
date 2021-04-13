#include <zen/zen.h>
#include <zen/MeshObject.h>
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
            float dt = std::get<float>(get_param("dt"));
            float vx = std::get<float>(get_param("vx"));
            float vy = std::get<float>(get_param("vy"));
            float vz = std::get<float>(get_param("vz"));
            auto velocity = get_input("Velocity")->as<VDBFloat3Grid>();
            auto face_weight = get_input("CellFWeight")->as<VDBFloat3Grid>();
            
            FLIP_vdb::field_add_vector(velocity->m_grid, 
            face_weight->m_grid, vx, vy, vz, dt);
        }
    };

static int defFieldAddVector = zen::defNodeClass<FieldAddVector>("FieldAddVector",
    { /* inputs: */ {
        "Velocity", "CellFWeight", 
    }, 
    /* outputs: */ {
    }, 
    /* params: */ {
       {"float", "dt", "0.0"},
       {"float", "vx", "0.0"},
       {"float", "vy", "0.0"},
       {"float", "vz", "0.0"},
    }, 
    
    /* category: */ {
    "FLIPSolver",
    }});

}