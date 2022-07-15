#include "FLIP_vdb.h"
#include <omp.h>
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>

namespace zeno {

struct rigid_body_surface_voxel_mask_reducer {

  rigid_body_surface_voxel_mask_reducer(
      const std::vector<zeno::vec3f> &inPosArray,
      openvdb::FloatGrid::Ptr inLiquidSDF)
      : myPosArray(inPosArray), myLiquidSDF(inLiquidSDF) {
    myMask = openvdb::BoolGrid::create(false);
  }

  rigid_body_surface_voxel_mask_reducer(
      const rigid_body_surface_voxel_mask_reducer &other, tbb::split)
      : myPosArray(other.myPosArray), myLiquidSDF(other.myLiquidSDF) {
    myMask = openvdb::BoolGrid::create(false);
  }

  void operator()(const tbb::blocked_range<size_t> &r) {
    using namespace openvdb::tools::local_util;
    // Const accessors
    auto LiquidSDFAccessor(myLiquidSDF->getConstUnsafeAccessor());
    auto MaskAccessor(myMask->getAccessor());

    // algorithm:
    // step1:
    // pos is a refined sample of object face,
    // so here simply rasterize it to a temple vdb field mask
    // mask = make vdbFloatGrid(0), transform(dx)
    // for p in pos, ijk = worldToIndex(p), mask(ijk)=1
    for (size_t i = r.begin(); i != r.end(); ++i) {
      const auto &p = myPosArray[i];
      // 1 find the index position of this position
      openvdb::Vec3f vdbwpos{p[0], p[1], p[2]};
      openvdb::Vec3f vdbipos = myLiquidSDF->worldToIndex(vdbwpos);

      // note phi is defined at the cell center
      // therefore to find the right sdf voxel this point belongs to,
      // a 0.5f shift is required.
      const openvdb::Coord gcoord{floorVec3(vdbipos + openvdb::Vec3f(0.5f))};
      if (LiquidSDFAccessor.getValue(gcoord) < 0) {
        MaskAccessor.setValueOn(gcoord);
      }
    } // End loop all points
  }

  void join(const rigid_body_surface_voxel_mask_reducer &other) {
    myMask->topologyUnion(*other.myMask);
  }

  const std::vector<zeno::vec3f> &myPosArray;

  openvdb::BoolGrid::Ptr myMask;
  openvdb::FloatGrid::Ptr myLiquidSDF;
};

struct rigid_pressure_force_reducer {
  rigid_pressure_force_reducer(
      const std::vector<openvdb::BoolTree::LeafNodeType *> &inMaskLeaves,
      const openvdb::Vec3f &inMassCenter, openvdb::FloatGrid::Ptr inPressure,
      openvdb::Vec3fGrid::Ptr inFaceWeights, float indt)
      : myMaskLeaves(inMaskLeaves), myMassCenter(inMassCenter),
        myPressure(inPressure), myFaceWeights(inFaceWeights), dt(indt) {
    myTotalForce = openvdb::Vec3f{0.f};
    myTotalTorque = openvdb::Vec3f{0.f};
  }

  rigid_pressure_force_reducer(const rigid_pressure_force_reducer &other,
                               tbb::split)
      : myMaskLeaves(other.myMaskLeaves), myMassCenter(other.myMassCenter),
        myPressure(other.myPressure), myFaceWeights(other.myFaceWeights),
        dt(other.dt) {
    myTotalForce = openvdb::Vec3f{0.f};
    myTotalTorque = openvdb::Vec3f{0.f};
  }

  void operator()(const tbb::blocked_range<size_t> &r) {
    using namespace openvdb::tools::local_util;
    // Const accessors
    auto PressureAccessor(myPressure->getConstUnsafeAccessor());
    auto FaceWeightsAccessor(myFaceWeights->getConstUnsafeAccessor());

    double dx = myPressure->voxelSize()[0];
    double dx2 = dx * dx;
    for (size_t i = r.begin(); i != r.end(); ++i) {
      const auto &leaf = *myMaskLeaves[i];

      // for all active liquid masks
      for (auto iter = leaf.beginValueOn(); iter; ++iter) {
        const auto gcoord = iter.getCoord();
        openvdb::Vec3f voxelCenter = myPressure->indexToWorld(gcoord);
        // step2:
        // mathematics: the pressure force a boundary element receives
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
        openvdb::Vec3f ForceToFluid{0.f};
        openvdb::Vec3f thisWeight = FaceWeightsAccessor.getValue(gcoord);
        ForceToFluid.x() =
            FaceWeightsAccessor.getValue(gcoord.offsetBy(1, 0, 0))[0] -
            thisWeight[0];
        ForceToFluid.y() =
            FaceWeightsAccessor.getValue(gcoord.offsetBy(0, 1, 0))[1] -
            thisWeight[1];
        ForceToFluid.z() =
            FaceWeightsAccessor.getValue(gcoord.offsetBy(0, 0, 1))[2] -
            thisWeight[2];
        openvdb::Vec3f ForceFromFluid =
            -ForceToFluid * dx2 * PressureAccessor.getValue(gcoord);
        // as a result, we only cumulate for each boundary
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
        myTotalTorque +=
            dt * (voxelCenter - myMassCenter).cross(ForceFromFluid);
      } // end for all active voxels
    }   // end for all leaves
  }     // End operator()

  void join(const rigid_pressure_force_reducer &other) {
    myTotalForce += other.myTotalForce;
    myTotalTorque += other.myTotalTorque;
  }

  const std::vector<openvdb::BoolTree::LeafNodeType *> &myMaskLeaves;
  const openvdb::Vec3f &myMassCenter;
  const float dt;

  openvdb::FloatGrid::Ptr myPressure;
  openvdb::Vec3fGrid::Ptr myFaceWeights;
  openvdb::Vec3f myTotalForce;
  openvdb::Vec3f myTotalTorque;
};
void samplePressureForce(float dt, std::vector<zeno::vec3f> &pos,
                         openvdb::FloatGrid::Ptr &pressure,
                         openvdb::Vec3fGrid::Ptr &faceWeights,
                         openvdb::FloatGrid::Ptr &liquid_sdf,
                         zeno::vec3f &massCenter, zeno::vec3f &totalForce,
                         zeno::vec3f &totalTorc) {
  // algorithm:
  // step1:
  // pos is a refined sample of object face,
  // so here simply rasterize it to a temple vdb field mask
  // mask = make vdbFloatGrid(0), transform(dx)
  // for p in pos, ijk = worldToIndex(p), mask(ijk)=1
  rigid_body_surface_voxel_mask_reducer mask_reducer(pos, liquid_sdf);

  // The grain size here prevent tbb generate too much threads because creating
  // a tree has overhead. Tune this parameter to get optimal performance
  tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, pos.size(), /*grain size*/ 1),
      mask_reducer);

  std::vector<openvdb::BoolTree::LeafNodeType *> mask_leaves;
  mask_leaves.reserve(mask_reducer.myMask->tree().leafCount());
  mask_reducer.myMask->tree().getNodes(mask_leaves);

  // step2:
  // mathematics: the pressure force a boundary element receives
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
  openvdb::Vec3f masscenter{massCenter[0], massCenter[1], massCenter[2]};
  rigid_pressure_force_reducer ForceReducer(mask_leaves, masscenter, pressure,
                                            faceWeights, dt);

  tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, mask_leaves.size(), /*grain size*/ 1),
      ForceReducer);

  for (int i = 0; i < 3; i++) {
    totalForce[i] = ForceReducer.myTotalForce[i];
    totalTorc[i] = ForceReducer.myTotalTorque[i];
  }
}
struct RigidGetPressureForce : zeno::INode {
  virtual void apply() override {
    auto dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
    auto massCenter =
        get_input("MassCenter")->as<zeno::NumericObject>()->get<zeno::vec3f>();
    auto rigid = get_input("Rigid")->as<zeno::PrimitiveObject>();
    auto Pressure = get_input("Pressure")->as<zeno::VDBFloatGrid>();
    auto CellFWeight = get_input("CellFWeight")->as<zeno::VDBFloat3Grid>();
    auto LiquidSDF = get_input("LiquidSDF")->as<zeno::VDBFloatGrid>();
    auto TotalForceImpulse = zeno::IObject::make<zeno::NumericObject>();
    auto TotalTorcImpulse = zeno::IObject::make<zeno::NumericObject>();
    std::vector<zeno::vec3f> pos;
    pos.resize(rigid->attr<zeno::vec3f>("pos").size());
#pragma omp parallel for
    for (int i = 0; i < rigid->attr<zeno::vec3f>("pos").size(); i++) {
      pos[i] = rigid->attr<zeno::vec3f>("pos")[i];
    }
    zeno::vec3f totalForce;
    zeno::vec3f totalTorc;
    samplePressureForce(dt, pos, Pressure->m_grid, CellFWeight->m_grid,
                        LiquidSDF->m_grid, massCenter, totalForce, totalTorc);
    TotalForceImpulse->set<zeno::vec3f>(totalForce);
    TotalTorcImpulse->set<zeno::vec3f>(totalTorc);
    set_output("TotalForceImpulse", TotalForceImpulse);
    set_output("TotalTorcImpulse", TotalTorcImpulse);
  }
};
static int defRigidGetPressureForce = zeno::defNodeClass<RigidGetPressureForce>(
    "RigidGetPressureForce", {/* inputs: */ {
                                  "dt",
                                  "MassCenter",
                                  "Rigid",
                                  "Pressure",
                                  "CellFWeight",
                                  "LiquidSDF",
                              },
                              /* outputs: */
                              {
                                  "TotalForceImpulse",
                                  "TotalTorcImpulse",
                              },
                              /* params: */
                              {
                                  {"float", "dx", "0.0"},
                              },

                              /* category: */
                              {
                                  "FLIPSolver",
                              }});
} // namespace zeno