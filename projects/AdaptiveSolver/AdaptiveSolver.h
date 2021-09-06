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

        void initData(){
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