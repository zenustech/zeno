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

#include <iostream>
namespace zeno{
    struct AdaptiveRule{
        virtual void markSubd(openvdb::FloatGrid::Ptr &grid, openvdb::FloatGrid::Ptr &tag){}
    };
    struct AdaptiveIndexGenerator{
        std::vector<openvdb::FloatGrid::Ptr> topoLevels;
        std::vector<openvdb::FloatGrid::Ptr> tag;
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

    struct LiquidAdaptiveRule : AdaptiveRule{
        virtual void markSubd(openvdb::FloatGrid::Ptr &grid, openvdb::FloatGrid::Ptr &tag) override
        {
            float dx = grid->voxelSize()[0];
            std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
            std::vector<openvdb::FloatTree::LeafNodeType *> tagleaves;
            openvdb::FloatGrid::Ptr tmpGrid;
            int markCount = 0, leaveCount = 0;
            auto extend = [&](const tbb::blocked_range<size_t> &r) {
                auto grid_axr{grid->getAccessor()};
                auto tag_axr{tag->getAccessor()};
                auto write_axr{tmpGrid->getAccessor()};
                // leaf iter
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *tagleaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        auto voxelwpos =
                            tag->indexToWorld(leaf.offsetToGlobalCoord(offset));
                        auto voxelipos = openvdb::Vec3i(tag->worldToIndex(voxelwpos));

                        if(tag->tree().isValueOff(openvdb::Coord(voxelipos)))
                            continue;
                        float value = tag_axr.getValue(openvdb::Coord(voxelipos));
                        // compute level set value far from the surface
                        if(value == -1)
                        {
                            float dis[3] = {100000,100000,100000};
                            int sign[3];
                            
                            for(int i=-1;i<=1;i += 2)
                            for(int select = 0;select < 3;++select)
                            {
                                auto base = openvdb::Vec3i(0,0,0);
                                base[select] = i;
                                auto ipos = voxelipos + base;
                                float nei_value = grid_axr.getValue(openvdb::Coord(ipos));
                                for(int t = 0;t < 3;++t)
                                    if(abs(nei_value) < dis[t] && nei_value != 0)
                                    {
                                        for(int tt= 2;tt>=t+1;--tt)
                                        {
                                            dis[tt] = dis[tt-1];
                                            sign[tt] = sign[tt-1];
                                        }
                                        dis[t] = abs(nei_value);
                                        sign[t] = nei_value / abs(nei_value);
                                        break;
                                    }
                            }
                            value = grid_axr.getValue(openvdb::Coord(voxelipos));
                            float d = dis[0] + dx;
                            if(d > dis[1])
                            {
                                d = 0.5 * (dis[0] + dis[1] + sqrt(2 * dx * dx - (dis[1]-dis[0]) * (dis[1]-dis[0])));
                                if(d > dis[2])
                                {
                                    float delta = dis[0] + dis[1] + dis[2];
                                    delta = delta * delta  - 3 *(dis[0] * dis[0] + 
                                        dis[1] * dis[1] + dis[2] * dis[2] - dx * dx);
                                    if(delta < 0)
                                        delta = 0;
                                    d = 0.3333 * (dis[0] + dis[1] + dis[2] + sqrt(delta));
                                }
                            }
                            if(d < abs(value) || value == 0)
                            {
                                //if no nei grid is valid, sign[0] equals zero, without bug here.
                                value = sign[0] * d;
                                write_axr.setValue(openvdb::Coord(voxelipos), value);
                            }
                        }
                    } // end for all on voxels
                }
            };
            
            // mark the points based on level set
            auto mark = [&](const tbb::blocked_range<size_t> &r) {
                auto grid_axr{grid->getAccessor()};
                auto tag_axr{tag->getAccessor()};
                // leaf iter
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        auto voxelwpos =
                            grid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                        auto voxelipos = openvdb::Vec3i(grid->worldToIndex(voxelwpos));

                        if(grid->tree().isValueOff(openvdb::Coord(voxelipos)))
                            continue;
                        float grid_value = grid_axr.getValue(openvdb::Coord(voxelipos));
                        tag_axr.setValue(openvdb::Coord(voxelipos), 1);
                        if(grid_value > dx || grid_value < - dx)
                            continue;
                        // if(grid_value == 0)
                        //     printf("grid value is zero on index (%d,%d,%d), pos (%f,%f,%f)\n",
                        //         voxelipos[0],voxelipos[1],voxelipos[2],
                        //         voxelwpos[0], voxelwpos[1], voxelwpos[2]);
                        for(int i=-1;i<=1;i += 1)
                        for(int j=-1;j<=1;j += 1)
                        for(int k=-1;k<=1;k += 1)
                        {
                            auto ipos = voxelipos + openvdb::Vec3i(i,j,k);
                            float value = grid_axr.getValue(openvdb::Coord(ipos));
                            if(grid->tree().isValueOff(openvdb::Coord(voxelipos)))
                            {
                                tag_axr.setValue(openvdb::Coord(ipos), -1);
                                markCount += 1;
                            }
                        }

                    } // end for all on voxels
                }
            };
            grid->tree().getNodes(leaves);
            printf("before mark,  grid active voxel is %d, tag active voxel is %d\n", 
                grid->activeVoxelCount(), tag->activeVoxelCount());

            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), mark);
            //LiquidAdaptiveRule::mark(grid, tag, markCount);
            printf("mark done, mark count is %d, grid active voxel is %d, tag active voxel is %d\n", 
                markCount, grid->activeVoxelCount(), tag->activeVoxelCount());
            tag->tree().getNodes(tagleaves);
            for(int count = 0;count < 3;++count)
            {
                tmpGrid = grid->deepCopy();
                //printf("begin to extend, grid size is %d\n", tagleaves.size());
                tbb::parallel_for(tbb::blocked_range<size_t>(0, tagleaves.size()), extend);
                //LiquidAdaptiveRule::extend(grid, tag, tmpGrid, leaveCount);
                printf("end to extend, leaveCount is %d, grid active voxel is %d, tag active voxel is %d\n",
                    leaveCount, grid->activeVoxelCount(), tag->activeVoxelCount());
                
                leaveCount = 0;
                grid = tmpGrid->deepCopy();
            }
            printf("extend done\n");
        };
    };

    struct TestRule : AdaptiveRule{
        //particle list
        //std::shared_ptr<zeno::PrimitiveObject> p;
        // refinement the details
        virtual void markSubd(openvdb::FloatGrid::Ptr &grid, openvdb::FloatGrid::Ptr &tag) override
        {
            //openvdb::FloatGrid::Ptr grid = aig.topoLevels[level];
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
            grid->tree().getNodes(leaves);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), mark);
            tmpGrid = grid->deepCopy();
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), extend);
            grid = tmpGrid->deepCopy();
        };
    
    };

}