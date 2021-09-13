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
    struct smokeData: IObject{
        // vertex attribute
        openvdb::FloatGrid::Ptr temperatureField;
        openvdb::FloatGrid::Ptr volumeField;

        // cell centered
        openvdb::FloatGrid::Ptr pressField;

        // face centered
        openvdb::FloatGrid::Ptr velField[3];

        // parms
        float dx, dt;
        openvdb::BBoxd worldinputbbox;
        void initData(openvdb::FloatGrid::Ptr sdf, float inputdt);

        void advection();
        void applyOuterforce();
        void solvePress();
        void step();
    };
};