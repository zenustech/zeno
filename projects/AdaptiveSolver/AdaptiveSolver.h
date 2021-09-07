#pragma once

#include "AdaptiveGridGen.h"
#include <openvdb/tools/ValueTransformer.h>


namespace zeno{
    // init the staggered press grid according to sdfGrid
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
    struct mgData{
        zeno::AdaptiveIndexGenerator aig;
        std::vector<openvdb::Vec3fGrid::Ptr> vel;
        std::vector<openvdb::FloatGrid::Ptr> press;

        // used for iteration solver
        std::vector<openvdb::FloatGrid::Ptr> rhs;
        std::vector<openvdb::FloatGrid::Ptr> residual;
        std::vector<openvdb::FloatGrid::Ptr> r2;
        std::vector<openvdb::FloatGrid::Ptr> p;
        std::vector<openvdb::FloatGrid::Ptr> Ap;

        void fillInner(){
            auto sdf = aig.topoLevels[0];
            auto dx = aig.hLevels[0];
            openvdb::FloatGrid::Ptr tag = zeno::IObject::make<VDBFloatGrid>()->m_grid;
            tag->setTree(std::make_shared<openvdb::FloatTree>(
                    sdf->tree(), /*bgval*/ float(1),
                    openvdb::TopologyCopy()));
            int activeNum = sdf->activeVoxelCount();
            std::vector<openvdb::FloatTree::LeafNodeType *> leaves;

            auto extendTag = [&](const tbb::blocked_range<size_t> &r){
                auto tag_axr{tag->getAccessor()};
                auto sdf_axr{sdf->getConstAccessor()};
                
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        auto coord = leaf.offsetToGlobalCoord(offset);
                        if(!sdf_axr.isValueOn(coord) || tag_axr.getValue(coord) == 2 )
                            continue;
                        if( sdf_axr.getValue(coord) >= 0)
                            continue;
                        int count = 0;
                        for(int i = -1;i<=1; i += 2)
                        for(int j = 0;j<3;++j)
                        {
                            auto neighbor = coord;
                            neighbor[j] += i;
                            if(!sdf_axr.isValueOn(neighbor))
                            {
                                //sdf2_axr.setValueOn(neighbor);
                                tag_axr.setValue(neighbor, 0.0f);
                                //auto tV = tag_axr.getValue(neighbor);
                                //printf("set new tag on (%d,%d,%d), value is %f\n",
                                //    neighbor[0],neighbor[1], neighbor[2], tV);
                            }
                            else
                                count++;
                        }
                        if(count == 6)
                            tag_axr.setValue(coord, 2);
                        else
                            tag_axr.setValue(coord, 1);
                    }
                }
            };
            auto computeSDF = [&](const tbb::blocked_range<size_t> &r){
                auto tag_axr{tag->getConstAccessor()};
                auto sdf_axr{sdf->getAccessor()};
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        auto coord = leaf.offsetToGlobalCoord(offset);
                        if(!tag_axr.isValueOn(coord) || tag_axr.getValue(coord) > 0.01f)
                        {
                            continue;
                        }
                        // printf("coord (%d,%d,%d) tag is %f\n", 
                        //     coord[0], coord[1], coord[2],
                        //     tag_axr.getValue(coord));
                        float dis[3] = {100000,100000,100000};
                        int sign[3];
                        
                        for(int i=-1;i<=1;i += 2)
                        for(int select = 0;select < 3;++select)
                        {
                            auto base = openvdb::Vec3i(0,0,0);
                            base[select] = i;
                            auto ipos = coord + base;
                            if(!tag_axr.isValueOn(openvdb::Coord(ipos)) || tag_axr.getValue(openvdb::Coord(ipos)) == 0)
                                continue;
                            float nei_value = sdf_axr.getValue(openvdb::Coord(ipos));
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
                        float value = sign[0] * d;
                        sdf_axr.setValue(coord, value);
                        
                        // printf("coord (%d,%d,%d) new value is %f\n", 
                        //     coord[0], coord[1], coord[2],
                        //     sign[0] * d);

                    }
                }
            };

            tag->tree().getNodes(leaves);
            for(int i=0;;++i)
            {
                tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), extendTag);
                leaves.clear();
                tag->tree().getNodes(leaves);
                //printf("leaves size is %d\n", leaves.size());
                tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeSDF);
                
                int newAN = sdf->activeVoxelCount();
                printf("activeNum is %d, newAN is %d\n", activeNum, newAN);
                if(activeNum == newAN)
                    break;
                else
                    activeNum = newAN;
            }
        }
        void initData(){
            // fill the coarsest level grid
            fillInner();
            for(int i=0;i<aig.hLevels.size();++i)
            {
                if(i > 0)
                {
                    aig.hLevels[i] = aig.hLevels[i-1] * 0.5;
                }
            }
            openvdb::FloatGrid::Ptr sdfgrid;
            openvdb::FloatGrid::Ptr pressGrid;
            for(int i=aig.hLevels.size() - 1;i >= 0;--i)
            {
                sdfgrid = aig.topoLevels[i];
                pressGrid = press[i] = sdfgrid->deepCopy();
                rhs[i] = sdfgrid->deepCopy();
                vel[i]->setTree(std::make_shared<openvdb::Vec3fTree>(
                    sdfgrid->tree(), /*bgval*/ openvdb::Vec3f(0),
                    openvdb::TopologyCopy()));
                auto transform2 = openvdb::math::Transform::createLinearTransform(aig.hLevels[i]);
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
            }
            
        }

        void resize(int levelNum)
        {
            aig.topoLevels.resize(levelNum);
            aig.hLevels.resize(levelNum);
            vel.resize(levelNum);
            press.resize(levelNum);
            rhs.resize(levelNum);
            residual.resize(levelNum);
            r2.resize(levelNum);
            p.resize(levelNum);
            Ap.resize(levelNum);
            for(int i=0;i < levelNum;++i)
            {
                //data.aig.topoLevels[i] = zeno::IObject::make<VDBFloatGrid>();
                vel[i] = zeno::IObject::make<VDBFloat3Grid>()->m_grid;
                press[i] = zeno::IObject::make<VDBFloatGrid>()->m_grid;
                rhs[i] = zeno::IObject::make<VDBFloatGrid>()->m_grid;
                residual[i] = zeno::IObject::make<VDBFloatGrid>()->m_grid;
                r2[i] = zeno::IObject::make<VDBFloatGrid>()->m_grid;
                p[i] = zeno::IObject::make<VDBFloatGrid>()->m_grid;
                Ap[i] = zeno::IObject::make<VDBFloatGrid>()->m_grid;

            }
        }
    
    };
}