#pragma once

#include <tbb/concurrent_vector.h>
#include <vector>
#include <zeno/zeno.h>
#include "tbb/scalable_allocator.h"
#include <cmath>
#include <zeno/ZenoInc.h>
#include "openvdb/points/PointConversion.h"
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <zeno/VDBGrid.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/MeshToVolume.h>
#include <iostream>
#include <openvdb/tools/LevelSetUtil.h>
#include <tbb/parallel_for.h>
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
        
    };
    struct agData:IObject{
        int levelNum;
        float dt, dens;
        
        openvdb::Int32Grid::Ptr tag;

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
        
        void PossionSolver();

        // multigrid methods
        void Vcycle();
        void Smooth(int level);
        void Restrict(int level);
        //void Propagate(int level);
        void GhostValueAccumulate(int level);
        void GhostValuePropagate(int level);
        void Prolongate(int level);

        // iterate functions
        void computeRHS(); //divergence of vel, store result on buffer.rhsGrid
        void computeLap(std::vector<openvdb::FloatGrid::Ptr> p, std::vector<openvdb::FloatGrid::Ptr> Ap);//div of grad p, store result on Ap
        void comptueRES();

        // normal functions
        float dotTree(std::vector<openvdb::FloatGrid::Ptr> a, std::vector<openvdb::FloatGrid::Ptr> b);
        // alpha * a + beta * b -> c
        void addTree(
            float alpha,float beta,
            std::vector<openvdb::FloatGrid::Ptr> a, 
            std::vector<openvdb::FloatGrid::Ptr> b,
            std::vector<openvdb::FloatGrid::Ptr> c);
        openvdb::Coord round(openvdb::Vec3d a){openvdb::Coord coord = openvdb::Coord(std::round(a[0]), std::round(a[1]),std::round(a[2]));return coord;}

        void makeCoarse();
        void transferPress(std::vector<openvdb::FloatGrid::Ptr> p);
        void applyPress();
    };
}