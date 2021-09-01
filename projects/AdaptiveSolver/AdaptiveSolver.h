#pragma once

#include "AdaptiveGridGen.h"


namespace zeno{
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
        
        void initData(){
            for(int i=0;i<aig.hLevels.size();++i)
            {
                if(i > 0)
                {
                    aig.hLevels[i] = aig.hLevels[i-1] * 0.5;
                }
                auto transform1 = openvdb::math::Transform::createLinearTransform(aig.hLevels[i]);
                auto transform2 = openvdb::math::Transform::createLinearTransform(aig.hLevels[i]);
                transform1->postTranslate(openvdb::Vec3d{0.5,0.5,0.5} *
                                                double(aig.hLevels[i]));
                vel[i]->setTransform(transform1);
                press[i]->setTransform(transform2);
                rhs[i]->setTransform(transform2);
                residual[i]->setTransform(transform2);
                r2[i]->setTransform(transform2);
                p[i]->setTransform(transform2);
                Ap[i]->setTransform(transform2);
            }
            openvdb::FloatGrid::Ptr sdfgrid;
            openvdb::Vec3fGrid::Ptr velGrid;
            openvdb::FloatGrid::Ptr pressGrid;
            openvdb::FloatGrid::Ptr rhsGrid;
            std::vector<openvdb::FloatTree::LeafNodeType *> leaves;

            // mark the points based on level set
            auto setzero = [&](const tbb::blocked_range<size_t> &r) {
                auto grid_axr{sdfgrid->getAccessor()};
                auto vel_axr{velGrid->getAccessor()};
                auto press_axr{pressGrid->getAccessor()};
                auto rhs_axr{rhsGrid->getAccessor()};

                // leaf iter
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        auto voxelwpos =
                            sdfgrid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                        auto voxelipos = openvdb::Vec3i(sdfgrid->worldToIndex(voxelwpos));

                        if(sdfgrid->tree().isValueOff(openvdb::Coord(voxelipos)))
                            continue;
                        vel_axr.setValue(openvdb::Coord(voxelipos), openvdb::Vec3f(0,0,0));
                        for(int i=0;i<=1;++i)
                        for(int j=0;j<=1;++j)
                        for(int k=0;k<=1;++k)
                        {
                            auto drift = voxelipos + openvdb::Vec3i(i,j,k);
                            press_axr.setValue(openvdb::Coord(drift), 0);
                            rhs_axr.setValue(openvdb::Coord(drift), 0);
                        }

                    }
                }
            };
            
            for(int i=aig.hLevels.size() - 1;i >= 0;--i)
            {
                sdfgrid = aig.topoLevels[i];
                velGrid = vel[i];
                pressGrid = press[i];
                rhsGrid = rhs[i];
                sdfgrid->tree().getNodes(leaves);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), setzero);
                leaves.clear();
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