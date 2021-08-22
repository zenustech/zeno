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

namespace zeno{

    struct AdaptiveRule{
        virtual void markSubd(openvdb::FloatGrid::Ptr &grid){}
        
    };
    struct LiquidAdaptiveRule : AdaptiveRule{
        //particle list
        //std::shared_ptr<zeno::PrimitiveObject> p;
        virtual void markSubd(openvdb::FloatGrid::Ptr &grid) override
        {
            
        }
    };
    struct TestRule : AdaptiveRule{
        //particle list
        //std::shared_ptr<zeno::PrimitiveObject> p;
        // refinement the details
        virtual void markSubd(openvdb::FloatGrid::Ptr &grid) override
        {
            float dx = grid->voxelSize()[0];
            std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
            openvdb::FloatGrid::Ptr tmpGrid;
            auto extend = [&](const tbb::blocked_range<size_t> &r) {
                auto grid_axr{grid->getAccessor()};
                auto write_axr{tmpGrid->getAccessor()};
                // leaf iter
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        auto voxelwpos =
                            grid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                        auto voxelipos = openvdb::Vec3i(grid->worldToIndex(voxelwpos));
                        float value = grid_axr.getValue(openvdb::Coord(voxelipos));
                        if(value == 1.0)
                        {
                            for(int i = -1;i<=1;i++)
                            {
                                for(int j=-1;j<=1;j++)
                                {
                                    for(int k=-1;k<=1;k++)
                                    {
                                        openvdb::Vec3i coord = voxelipos+openvdb::Vec3i(i,j,k);
                                        write_axr.setValue(openvdb::Coord(coord), 1.0);
                                    }
                                }
                            }
                        }
                    } // end for all on voxels
                }
            };
            // mark the points based on pos
            auto mark = [&](const tbb::blocked_range<size_t> &r) {
                auto grid_axr{grid->getAccessor()};
                
                // leaf iter
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        auto voxelwpos =
                            grid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                        auto voxelipos = openvdb::Vec3i(grid->worldToIndex(voxelwpos));
                        float value =  std::sqrt(voxelwpos[0] * voxelwpos[0] + voxelwpos[1]*voxelwpos[1] + voxelwpos[2]*voxelwpos[2]) - 1.0f;
                        if(std::abs(value)<1.01f*std::sqrt(3.0f)*dx)
                        {
                            grid_axr.setValue(openvdb::Coord(voxelipos), 1.0);
                        }
                    } // end for all on voxels
                }
            };
            // grid->tree().getNodes(leaves);
            // tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), mark);
            // tmpGrid = grid->deepCopy();
            // tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), extend);
            // grid = tmpGrid->deepCopy();
        }
    };
    struct AdaptiveIndexGenerator{
    std::vector<openvdb::FloatGrid::Ptr> topoLevels;
    std::vector<double> hLevels;
    void generateAdaptiveGrid(
        AdaptiveIndexGenerator& data, 
        int max_levels, 
        double start_h,
        std::shared_ptr<AdaptiveRule> rule
        );
    openvdb::Vec3d abs(openvdb::Vec3d a)
    {
        openvdb::Vec3d b;
        for(int i=0;i<3;++i)
            if(a[i]<0)
                b[i] = -a[i];
            else
                b[i] = a[i];
        return b;
    };
    };

}