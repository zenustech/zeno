#include <stdio.h>
#include <stdlib.h>
#include <openvdb/openvdb.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/points/PointConversion.h>

openvdb::points::PointDataGrid::Ptr particleArrayToGrid(std::vector<openvdb::Vec3f>& positions, std::vector<openvdb::Vec3f> &velocitys, float dx) 
{
    // The VDB Point-Partioner is used when bucketing points and requires a
    // specific interface. For convenience, we use the PointAttributeVector
    // wrapper around an stl vector wrapper here, however it is also possible to
    // write one for a custom data structure in order to match the interface
    // required.
    openvdb::points::PointAttributeVector<openvdb::Vec3f> positionsWrapper(positions);
    // This method computes a voxel-size to match the number of
    // points / voxel requested. Although it won't be exact, it typically offers
    // a good balance of memory against performance.
    int pointsPerVoxel = 8;
    float voxelSize = dx;
    // Print the voxel-size to cout
    // Create a transform using this voxel-size.
    openvdb::math::Transform::Ptr transform
        = openvdb::math::Transform::createLinearTransform(voxelSize);


    openvdb::tools::PointIndexGrid::Ptr pointIndexGrid =
        openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(
            positionsWrapper, *transform);
    // Create a PointDataGrid containing these four points and using the
    // transform given. This function has two template parameters, (1) the codec
    // to use for storing the position, (2) the grid we want to create
    // (ie a PointDataGrid).
    // We use no compression here for the positions.
    using PositionCodec = openvdb::points::FixedPointCodec</*one byte*/false>;
	//using PositionCodec = openvdb::points::TruncateCodec;
	//using PositionCodec = openvdb::points::NullCodec;
	using position_attribute = openvdb::points::TypedAttributeArray<openvdb::Vec3f, PositionCodec>;
	using VelocityCodec = openvdb::points::TruncateCodec;
	//using VelocityCodec = openvdb::points::NullCodec;
	using velocity_attribute = openvdb::points::TypedAttributeArray<openvdb::Vec3f, VelocityCodec>;
    auto pnamepair = position_attribute::attributeType();
    auto vnamepair = velocity_attribute::attributeType();

    openvdb::points::PointDataGrid::Ptr grid =
        openvdb::points::createPointDataGrid<PositionCodec,
            openvdb::points::PointDataGrid>(*pointIndexGrid, positionsWrapper, *transform);


    openvdb::points::appendAttribute(grid->tree(), "v", vnamepair);

    openvdb::points::PointAttributeVector<openvdb::Vec3f> velocityWrapper(velocitys);

    openvdb::points::populateAttribute<openvdb::points::PointDataTree,
        openvdb::tools::PointIndexTree, openvdb::points::PointAttributeVector<openvdb::Vec3f>>(
            grid->tree(), pointIndexGrid->tree(), "v", velocityWrapper);
    
    grid->setName("Points");
    return grid;
}
struct VDBGrid {
    virtual void input() = 0;
};
template <class T>
struct VDBWrapper : VDBGrid {
    typename T::Ptr m_grid;
    virtual void input() override { printf("input\n");; }
    VDBWrapper() { printf("ctor\n"); }
    ~VDBWrapper() { printf("dtor\n"); }
};

std::unique_ptr<VDBWrapper<openvdb::points::PointDataGrid>> wrap;

void createALotVDB(std::vector<openvdb::Vec3f>& positions, std::vector<openvdb::Vec3f> &velocitys, float dx)
{
    wrap = std::make_unique<VDBWrapper<openvdb::points::PointDataGrid>>();
        wrap->m_grid = particleArrayToGrid(positions, velocitys, dx);
}
int main(int argc, char* argv[])
{
    openvdb::initialize();
    //need a large enought point data...
    std::vector<openvdb::Vec3f> points;
    std::vector<openvdb::Vec3f> velocities;
    points.resize(512*512*512);
    velocities.resize(points.size());
    for(int i=0;i<points.size();i++)
    {
        points[i] = openvdb::Vec3f((float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX);
        velocities[i] = openvdb::Vec3f((float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX);
    }
    for(int i=0;i<100;i++)
    {
        createALotVDB(points, velocities, 1.0/512.0);
    }
    getchar();
    return 0;
}