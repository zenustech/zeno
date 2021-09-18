#pragma once
#include <tbb/concurrent_vector.h>
#include <vector>
#include <zeno/zeno.h>
#include "tbb/scalable_allocator.h"

#include <zeno/ZenoInc.h>
#include "openvdb/points/PointConversion.h"
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <zeno/VDBGrid.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/MeshToVolume.h>
#include <iostream>

namespace zeno{
    struct mgIterStuff{
        // iteration terms
        std::vector<openvdb::FloatGrid::Ptr> rhsGrid;
        std::vector<openvdb::FloatGrid::Ptr> resGrid;
        std::vector<openvdb::FloatGrid::Ptr> r2Grid;

        std::vector<openvdb::FloatGrid::Ptr> pGrid;
        std::vector<openvdb::FloatGrid::Ptr> ApGrid;
        void init(std::vector<openvdb::FloatGrid::Ptr> pressField);
        void resize(int levelNum);
    };
    struct mgSmokeData:IObject{
        using ConstBoxSample = openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>;
        using BoxSample = openvdb::tools::GridSampler<openvdb::FloatGrid::Accessor, openvdb::tools::BoxSampler>;

        // vertex attribute
        std::vector<openvdb::FloatGrid::Ptr> temperatureField;
        std::vector<openvdb::FloatGrid::Ptr> volumeField;
        openvdb::Int32Grid::Ptr tag;

        // cell centered
        std::vector<openvdb::FloatGrid::Ptr>  pressField;
        mgIterStuff iterBuffer;
        // face centered
        std::vector<openvdb::FloatGrid::Ptr>  velField[3];

        // parms
        float dt;
        std::vector<float> dx;
        openvdb::BBoxd worldinputbbox;
        void resize(int levelNum);
        void initData(openvdb::FloatGrid::Ptr sdf, int levelNum, float inputdt);

        void advection();
        void applyOuterforce();
        void solvePress();
        void step();
    };
}