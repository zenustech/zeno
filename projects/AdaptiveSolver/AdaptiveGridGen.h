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
    using ConstBoxSample = openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler>;
    using BoxSample = openvdb::tools::GridSampler<openvdb::FloatGrid::Accessor, openvdb::tools::BoxSampler>;
    struct agIterStuff{
        std::vector<openvdb::FloatGrid::Ptr> rhsGrid;
        std::vector<openvdb::FloatGrid::Ptr> resGrid;
        std::vector<openvdb::FloatGrid::Ptr> r2Grid;

        std::vector<openvdb::FloatGrid::Ptr> pGrid;
        std::vector<openvdb::FloatGrid::Ptr> ApGrid;
        void init(std::vector<openvdb::FloatGrid::Ptr> pressField);
        void resize(int levelNum);
        void computeRHSgrid(int level);
    };
    struct agData:IObject{
        int levelNum;
        float dt, dens;
        
        std::vector<float> dx;
        std::vector<openvdb::Int32Grid::Ptr> status;//0:active 1:ghost 2:inactive
        std::vector<openvdb::FloatGrid::Ptr> volumeField;
        std::vector<openvdb::FloatGrid::Ptr> temperatureField;
        std::vector<openvdb::FloatGrid::Ptr> velField[3];
        std::vector<openvdb::FloatGrid::Ptr> pressField;
        std::vector<openvdb::FloatGrid::Ptr> gradPressField[3];
        
        agIterStuff buffer;
        void step();
        void Advection();
        void applyOtherForce();
        void solvePress();
        void markGhost();

        void resize(int lNum);
        void initData(openvdb::FloatGrid::Ptr sdf, int lNum, float inputdt);
        
        void Vcycle();
        
        void PossionSolver(int level);

        // multigrid methods
        void Smooth(int level);
        void Restrict(int level);
        void Propagate(int level);
        void GhostValueAccumulate(int level);
        void GhostValuePropagate(int level);
        void Prolongate(int level);

        // iterate functions
        void computeRHS(int level); //divergence of vel, store result on buffer.rhsGrid
        void computeGradP(int level, openvdb::FloatGrid::Ptr p);//gradient of press, store result on buffer.gradp 
        void computeDivP(int level);//div of grad press, store result on buffer.ApGrid
        void comptueRES(int level, openvdb::FloatGrid::Ptr p);

        // normal functions
        float dotTree(int level, openvdb::FloatGrid::Ptr a, openvdb::FloatGrid::Ptr b);
    };
}