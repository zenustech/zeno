#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/points/PointCount.h>
#include "FLIP_vdb.h"

int main(int argc, char* argv[])
{
    openvdb::initialize();
    openvdb::points::PointDataGrid::Ptr particles;
    particles= openvdb::points::PointDataGrid::create();
    particles->setTransform(openvdb::math::Transform::createLinearTransform(0.1)); 
    particles->setName("Particles");
    openvdb::FloatGrid::Ptr sphereGrid =
    openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(/*radius=*/1.0,
            /*center=*/openvdb::Vec3f(0, 0, 0), /*voxel size=*/0.1);

    FLIP_vdb::emit_liquid(particles, sphereGrid, nullptr, 0,0,0);

    return 0;
}