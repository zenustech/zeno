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
       struct initializer{
        openvdb::FloatGrid::Ptr sdfGrid;
        openvdb::FloatGrid::Ptr pGrid;

        initializer(openvdb::FloatGrid::Ptr sdf, openvdb::FloatGrid::Ptr p):sdfGrid(sdf), pGrid(p){}
        
        inline void operator()(const openvdb::FloatGrid::ValueOnIter& sdfiter) const 
        {
            auto press_axr{pGrid->getAccessor()};
            openvdb::Coord coord = sdfiter.getCoord();
            for(int i=0;i<=1;++i)
            for(int j=0;j<=1;++j)
            for(int k=0;k<=1;++k)
            {
                openvdb::Coord drift = coord + openvdb::Coord(i,j,k);
                press_axr.setValue(drift, 0.0f);
            }
        }
    };
    struct mgData:IObject{
        std::vector<openvdb::FloatGrid::Ptr> sdf;
        std::vector<double> hLevels;

        std::vector<openvdb::Vec3fGrid::Ptr> vel;
        std::vector<openvdb::FloatGrid::Ptr> press;
        std::vector<openvdb::FloatGrid::Ptr> staggeredSDF;
        
        // used for iteration solver
        std::vector<openvdb::FloatGrid::Ptr> rhs;
        std::vector<openvdb::FloatGrid::Ptr> residual;
        std::vector<openvdb::FloatGrid::Ptr> r2;
        std::vector<openvdb::FloatGrid::Ptr> p;
        std::vector<openvdb::FloatGrid::Ptr> Ap;

        void initData(){
            // fill the coarsest level grid
            // fillInner();
            for(int i=0;i<hLevels.size();++i)
            {
                if(i > 0)
                {
                    hLevels[i] = hLevels[i-1] * 0.5;
                }
            }
            openvdb::FloatGrid::Ptr sdfgrid;
            openvdb::FloatGrid::Ptr pressGrid;
            for(int i=hLevels.size() - 1;i >= 0;--i)
            {
                sdfgrid = sdf[i];
                pressGrid = press[i] = sdfgrid->deepCopy();
                rhs[i] = sdfgrid->deepCopy();
                vel[i]->setTree(std::make_shared<openvdb::Vec3fTree>(
                    sdfgrid->tree(), /*bgval*/ openvdb::Vec3f(0),
                    openvdb::TopologyCopy()));
                auto transform2 = openvdb::math::Transform::createLinearTransform(hLevels[i]);
                press[i]->setTransform(transform2);
                
                openvdb::tools::foreach(sdfgrid->beginValueOn(), initializer(sdfgrid, press[i]));
                rhs[i]->setTree(std::make_shared<openvdb::FloatTree>(
                    pressGrid->tree(), /*bgval*/ float(0),
                    openvdb::TopologyCopy()));
                residual[i]->setTree(std::make_shared<openvdb::FloatTree>(
                    pressGrid->tree(), /*bgval*/ float(0),
                    openvdb::TopologyCopy()));
                r2[i]->setTree(std::make_shared<openvdb::FloatTree>(
                    pressGrid->tree(), /*bgval*/ float(0),
                    openvdb::TopologyCopy()));
                p[i]->setTree(std::make_shared<openvdb::FloatTree>(
                    pressGrid->tree(), /*bgval*/ float(0),
                    openvdb::TopologyCopy()));
                Ap[i]->setTree(std::make_shared<openvdb::FloatTree>(
                    pressGrid->tree(), /*bgval*/ float(0),
                    openvdb::TopologyCopy()));
                staggeredSDF[i]->setTree(std::make_shared<openvdb::FloatTree>(
                    pressGrid->tree(), /*bgval*/ float(0),
                    openvdb::TopologyCopy()));
            }
            
        }

        void resize(int levelNum)
        {
            sdf.resize(levelNum);
            hLevels.resize(levelNum);
            vel.resize(levelNum);
            press.resize(levelNum);
            staggeredSDF.resize(levelNum);
            rhs.resize(levelNum);
            residual.resize(levelNum);
            r2.resize(levelNum);
            p.resize(levelNum);
            Ap.resize(levelNum);
            for(int i=0;i < levelNum;++i)
            {
                vel[i] = zeno::IObject::make<VDBFloat3Grid>()->m_grid;
                press[i] = zeno::IObject::make<VDBFloatGrid>()->m_grid;
                staggeredSDF[i] = zeno::IObject::make<VDBFloatGrid>()->m_grid;
                rhs[i] = zeno::IObject::make<VDBFloatGrid>()->m_grid;
                residual[i] = zeno::IObject::make<VDBFloatGrid>()->m_grid;
                r2[i] = zeno::IObject::make<VDBFloatGrid>()->m_grid;
                p[i] = zeno::IObject::make<VDBFloatGrid>()->m_grid;
                Ap[i] = zeno::IObject::make<VDBFloatGrid>()->m_grid;
            }
        }
    
    };
    
    struct AdaptiveRule{
        virtual void markSubd(openvdb::FloatGrid::Ptr &grid, openvdb::FloatGrid::Ptr &tag){}
    };
    struct AdaptiveIndexGenerator{
        std::vector<openvdb::FloatGrid::Ptr> sdf;
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
            //openvdb::FloatGrid::Ptr grid = aig.sdf[level];
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