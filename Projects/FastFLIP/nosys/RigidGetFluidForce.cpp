#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <zen/NumericObject.h>
#include <zen/VDBGrid.h>
#include <omp.h>
#include "FLIP_vdb.h"




namespace zen{

    void samplePressureForce(
        float dt,
        std::vector<zen::vec3f> & pos,
        openvdb::FloatGrid::Ptr & pressure,
        openvdb::Vec3fGrid::Ptr & faceWeights,
        openvdb::FloatGrid::Ptr & liquid_sdf,
        zen::vec3f &massCenter,
        zen::vec3f &totalForce,
        zen::vec3f &totalTorc)
    {
        //algorithm:
        //step1:
        //pos is a refined sample of object face,
        //so here simply rasterize it to a temple vdb field mask
        //mask = make vdbFloatGrid(0), transform(dx)
        //for p in pos, ijk = worldToIndex(p), mask(ijk)=1

        //step2:
        //mathematics: the pressure force a boundary element receives
        // equals to  the pressure force it sends to fluid
        //         
        //              vweight(j+1)
        //             ____________
        //             |////////  |
        //             |/s///     |
        //  uweight(i) |///  p    | uweight(i+1)
        //             |/         |
        //             |__________|
        //              vweight(j)
        //
        //  ForceToFluid = dx^2*(
        //                        p*uweight(i+1)*(+1,0) + p*uweight(i)*(-1,0)
        //                       +p*vweight(j+1)*(0,+1) + p*vweight(j)*(0,-1)
        //                      )
        //  
        //  ForceFromFluid = - ForceToFluid;               Eqn 1
        //   note how this is a better approx to p*n*dA;
        //
        //  as a result, we only cumulate for each boundary 
        //  cell(we marked out in step1)
        //  
        //   atomic force = 0, torc = 0;
        //   for each marked boundary cell, 
        //      if liquid_phi<0
        //         force_of_the_cell = Eqn1
        //         force += force_of_the_cell;
        //         torc += cross(masscenter - pcell, force_of_the_cell);
        //
        // totalForce = dt*force
        // totalTorc = dt*torc;

    }
    struct RigidGetPressureForce:zen::INode{
        virtual void apply() override {

        }
    };
    static int defRigidGetPressureForce = zen::defNodeClass<RigidGetPressureForce>("RigidGetPressureForce",
    { /* inputs: */ {
        "dt", "MassCenter", "Rigid", "Pressure", "CellFWeight", "LiquidSDF",
    }, 
    /* outputs: */ {
        "TotalForceImpulse", "TotalTorcImpulse",
    }, 
    /* params: */ {
       {"float", "dx", "0.0"},
    }, 
    
    /* category: */ {
    "FLIPSolver",
    }});
}