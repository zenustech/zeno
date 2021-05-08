#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <zen/NumericObject.h>
#include <zen/VDBGrid.h>
#include <omp.h>
#include "FLIP_vdb.h"
#include "zen/VDBGrid.h"




namespace zenbase{
    
    struct CFL : zen::INode{
        virtual void apply() override {
            auto velocity = get_input("Velocity")->as<VDBFloat3Grid>();
            float dx = std::get<float>(get_param("dx"));
	    float dt = FLIP_vdb::cfl(velocity->m_grid);
            printf("CFL dt: %f\n", dt);
            auto out_dt = zen::IObject::make<zenbase::NumericObject>();
            out_dt->set<float>(dt);
            set_output("cfl_dt", out_dt);
        }
    };

static int defCFL = zen::defNodeClass<CFL>("CFL_dt",
    { /* inputs: */ {
        "Velocity", 
    }, 
    /* outputs: */ {
        "cfl_dt",
    }, 
    /* params: */ {
        {"float", "dx", "0.0"},
    }, 
    
    /* category: */ {
    "FLIPSolver",
    }});

}
