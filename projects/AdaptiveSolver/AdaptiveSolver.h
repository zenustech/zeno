#pragma once

#include "AdaptiveGridGen.h"
#include <openvdb/tools/ValueTransformer.h>


namespace zeno{
    // init the staggered press grid according to sdfGrid
    struct initializer{
        openvdb::FloatGrid::Ptr sdfGrid;
        openvdb::FloatGrid::Ptr pGrid;
        double spacing;
        initializer(openvdb::FloatGrid::Ptr sdf, openvdb::FloatGrid::Ptr p, double h):sdfGrid(sdf), pGrid(p), spacing(h){}
        
        void operator()(const openvdb::FloatGrid::ValueOnCIter& sdfiter) const 
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

        void initData(){
            for(int i=0;i<aig.hLevels.size();++i)
            {
                if(i > 0)
                {
                    aig.hLevels[i] = aig.hLevels[i-1] * 0.5;
                }
            }
            openvdb::FloatGrid::Ptr sdfgrid;
            openvdb::Vec3fGrid::Ptr velGrid;
            openvdb::FloatGrid::Ptr pressGrid;
            openvdb::FloatGrid::Ptr rhsGrid;
            std::vector<openvdb::FloatTree::LeafNodeType *> leaves;

            for(int i=aig.hLevels.size() - 1;i >= 0;--i)
            {
                sdfgrid = aig.topoLevels[i];
                velGrid = vel[i];
                
                pressGrid = press[i] = sdfgrid->deepCopy();
                rhsGrid = rhs[i] = sdfgrid->deepCopy();
                sdfgrid->tree().getNodes(leaves);
                vel[i]->setTree(std::make_shared<openvdb::Vec3fTree>(
                    sdfgrid->tree(), /*bgval*/ openvdb::Vec3f(0),
                    openvdb::TopologyCopy()));
                auto transform2 = openvdb::math::Transform::createLinearTransform(aig.hLevels[i]);
                press[i]->setTransform(transform2);
                rhs[i]->setTransform(transform2);
                #pragma omp parallel for
                for(int ii= 0 ;ii < sdfgrid->tree().leafCount(); ++ii)
                {
                    auto grid_axr{sdfgrid->getAccessor()};
                    auto press_axr{pressGrid->getAccessor()};
                    auto rhs_axr{rhsGrid->getAccessor()};
                    openvdb::FloatGrid::TreeType::LeafIter iter = sdfgrid->tree().beginLeaf();
                    for(int jj = 0;jj<ii;++jj)
                        ++iter;
                    
                    openvdb::FloatGrid::TreeType::LeafNodeType& leaf = *iter;
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        auto voxelwpos =
                            sdfgrid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                        auto voxelipos = openvdb::Vec3i(sdfgrid->worldToIndex(voxelwpos));

                        if(sdfgrid->tree().isValueOff(openvdb::Coord(voxelipos)))
                            continue;
                        press_axr.setValue(openvdb::Coord(voxelipos), 0.0f);
                        rhs_axr.setValue(openvdb::Coord(voxelipos), 0.0f);
                        
                        for(int i=0;i<=1;++i)
                        for(int j=0;j<=1;++j)
                        for(int k=0;k<=1;++k)
                        {
                            openvdb::Vec3i drift = voxelipos + openvdb::Vec3i(i,j,k);
                            if(sdfgrid->tree().isValueOn(openvdb::Coord(drift)))
                                continue;
                            
                            press_axr.setValue(openvdb::Coord(drift), 0.0f);
                            rhs_axr.setValue(openvdb::Coord(drift), 0.0f);
                        }

                    }
                }
                leaves.clear();

                openvdb::tools::foreach(pressGrid->beginValueOn(), setzero);
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