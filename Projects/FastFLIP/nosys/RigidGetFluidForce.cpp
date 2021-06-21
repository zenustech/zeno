#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <zen/NumericObject.h>
#include <zen/VDBGrid.h>
#include <omp.h>
#include "FLIP_vdb.h"




namespace zen{
struct rigid_pressure_force_reducer {
    rigid_pressure_force_reducer(const std::vector<zen::vec3f>& inPosArray,
        const openvdb::Vec3f& inMassCenter,
        openvdb::FloatGrid::Ptr inPressure,
        openvdb::Vec3fGrid::Ptr inFaceWeights,
        openvdb::FloatGrid::Ptr inLiquidSDF, float indt):
        myPosArray(inPosArray),
        myMassCenter(inMassCenter),
        myPressure(inPressure),
        myFaceWeights(inFaceWeights), 
        myLiquidSDF(inLiquidSDF),
        dt(indt) {
        myTotalForce = openvdb::Vec3f{ 0.f};
        myTotalTorque = openvdb::Vec3f{ 0.f };
        myMask = openvdb::BoolGrid::create(false);
    }

    rigid_pressure_force_reducer(const rigid_pressure_force_reducer& other, tbb::split):
        myPosArray(other.myPosArray), 
        myMassCenter(other.myMassCenter), 
        myPressure(other.myPressure),
        myFaceWeights(other.myFaceWeights),
        myLiquidSDF(other.myLiquidSDF),
        dt(other.dt) {
        myTotalForce = openvdb::Vec3f{ 0.f };
        myTotalTorque = openvdb::Vec3f{ 0.f };
        myMask = openvdb::BoolGrid::create(false);
    }

    void operator()(const tbb::blocked_range<size_t>& r) {
        using  namespace openvdb::tools::local_util;
        //Const accessors
        auto PressureAccessor(myPressure->getConstUnsafeAccessor());
        auto FaceWeightsAccessor(myFaceWeights->getConstUnsafeAccessor());
        auto LiquidSDFAccessor(myLiquidSDF->getConstUnsafeAccessor());
        auto MaskAccessor(myMask->getAccessor());

        double dx = myPressure->voxelSize()[0];
        double dx2 = dx * dx;
        for (size_t i = r.begin(); i != r.end(); ++i) {
            auto& p = myPosArray[i];
            //1 find the index position of this position
            openvdb::Vec3f vdbwpos{ p[0],p[1],p[2] };
            openvdb::Vec3f vdbipos = myPressure->worldToIndex(vdbwpos);

            //note p is defined at the cell center
            //therefore to find the right pressure voxel this point belongs to,
            //a 0.5f shift is required.
            const openvdb::Coord gcoord{ floorVec3(vdbipos + openvdb::Vec3f(0.5f)) };

            //skip a cell that has already contributed
            if (MaskAccessor.isValueOn(gcoord)) {
                continue;
            }

            //skip points outside of the liquid
            //this is tested after the mask because sampling can be more expensive
            if (openvdb::tools::BoxSampler::sample(LiquidSDFAccessor, vdbipos) >= 0) {
                continue;
            }

            MaskAccessor.setValueOn(gcoord);

            openvdb::Vec3f voxelCenter = myPressure->indexToWorld(gcoord);
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
            openvdb::Vec3f ForceToFluid{ 0.f };
            openvdb::Vec3f thisWeight = FaceWeightsAccessor.getValue(gcoord);
            ForceToFluid.x() = FaceWeightsAccessor.getValue(gcoord.offsetBy(1, 0, 0))[0] - thisWeight[0];
            ForceToFluid.y() = FaceWeightsAccessor.getValue(gcoord.offsetBy(0, 1, 0))[1] - thisWeight[1];
            ForceToFluid.z() = FaceWeightsAccessor.getValue(gcoord.offsetBy(0, 0, 1))[2] - thisWeight[2];
            openvdb::Vec3f ForceFromFluid = -ForceToFluid * dx2 * PressureAccessor.getValue(gcoord);
            //as a result, we only cumulate for each boundary
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
            myTotalForce += dt * ForceFromFluid;
            myTotalTorque += dt * (myMassCenter - voxelCenter).cross(ForceFromFluid);
        }//End loop all points
    }//End operator()

    void join(const rigid_pressure_force_reducer& other) {
        myTotalForce += other.myTotalForce;
        myTotalTorque += other.myTotalTorque;
        myMask->topologyUnion(*other.myMask);
    }


    const std::vector<zen::vec3f>& myPosArray;
    const openvdb::Vec3f& myMassCenter;
    const float dt;

    openvdb::BoolGrid::Ptr myMask;
    openvdb::FloatGrid::Ptr myPressure;
    openvdb::Vec3fGrid::Ptr myFaceWeights;
    openvdb::FloatGrid::Ptr myLiquidSDF;
    openvdb::Vec3f myTotalForce;
    openvdb::Vec3f myTotalTorque;
};
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

        openvdb::Vec3f masscenter{ massCenter[0],massCenter[1],massCenter[2] };
        rigid_pressure_force_reducer reducer(pos,
            masscenter,
            pressure,
            faceWeights,
            liquid_sdf, dt);

        //The grain size here prevent tbb generate too much threads because creating a tree has overhead.
        //Tune this parameter to get optimal performance
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, pos.size(),/*grain size*/100), reducer);
        for (int i = 0; i < 3; i++) {
            totalForce[i] = reducer.myTotalForce[i];
            totalTorc[i] = reducer.myTotalTorque[i];
        }

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