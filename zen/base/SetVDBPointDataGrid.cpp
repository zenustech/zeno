#if 0
#pragma once

#include <zen/zen.h>
#include <vector>

#include <openvdb/points/PointCount.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/MeshToVolume.h>
#include <zen/ParticlesObject.h>
openvdb::points::PointDataGrid::Ptr particleArrayToGrid(const ParticlesObject& particles) 
{
    std::vector<openvdb::Vec3R> positions(particles.size());
    std::vector<openvdb::Vec3R> velocitys(particles.size());
    // for (auto &&[dst, src] : zip(positions, particles))
    for (auto i = 0; i < particles.size(); ++i){
      for (int d = 0; d < 3; ++d) {
            positions[i][d] = particles.pos[i][d];
            velocitys[i][d] = particles.vel[i][d];
          
          }
    }
    // The VDB Point-Partioner is used when bucketing points and requires a
    // specific interface. For convenience, we use the PointAttributeVector
    // wrapper around an stl vector wrapper here, however it is also possible to
    // write one for a custom data structure in order to match the interface
    // required.
    openvdb::points::PointAttributeVector<openvdb::Vec3R> positionsWrapper(positions);
    // This method computes a voxel-size to match the number of
    // points / voxel requested. Although it won't be exact, it typically offers
    // a good balance of memory against performance.
    int pointsPerVoxel = 8;
    float voxelSize = openvdb::points::computeVoxelSize(positionsWrapper, pointsPerVoxel);
    // Print the voxel-size to cout
    // Create a transform using this voxel-size.
    openvdb::math::Transform::Ptr transform
        = openvdb::math::Transform::createLinearTransform(voxelSize);

    // Create a PointDataGrid containing these four points and using the
    // transform given. This function has two template parameters, (1) the codec
    // to use for storing the position, (2) the grid we want to create
    // (ie a PointDataGrid).
    // We use no compression here for the positions.
    using PDGPtr = openvdb::points::PointDataGrid::Ptr;
    PDGPtr ret = openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
                                                             openvdb::points::PointDataGrid>(
        positions, *transform);
    ret->setName("Points");
    return ret;
}

namespace zenbase {

struct SetVDBPointDataGrid : zen::INode {
  virtual void apply() override {
    
  }
};

static int defSetVDBPointDataGrid = zen::defNodeClass<SetVDBPointDataGrid>("SetVDBPointDataGrid",
    { /* inputs: */ {
        "ParticleGeo", 
    }, /* outputs: */ {
        "Particles",
    }, /* params: */ {
    {"float", "dx", "0.0"},
    }, /* category: */ {
        "particles",
    }});

}

#endif