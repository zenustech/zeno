#include "mgSmokeSolver.h"
#include <openvdb/tools/Interpolation.h>
#include "tbb/blocked_range3d.h"
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/FastSweeping.h>
#include "tbb/atomic.h"
namespace zeno{
    void mgIterStuff::init(std::vector<openvdb::FloatGrid::Ptr> pressField)
    {
        for(int i=0;i<pressField.size();++i)
        {
            rhsGrid[i] = pressField[i]->deepCopy();
            resGrid[i] = pressField[i]->deepCopy();
            r2Grid[i] = pressField[i]->deepCopy();
            pGrid[i] = pressField[i]->deepCopy();
            ApGrid[i] = pressField[i]->deepCopy();
        }
    }
    void mgIterStuff::resize(int levelNum)
    {
        rhsGrid.resize(levelNum);
        resGrid.resize(levelNum);
        r2Grid.resize(levelNum); 
        pGrid.resize(levelNum);
        ApGrid.resize(levelNum);
        for(int i=0;i<levelNum;++i)
        {
            rhsGrid[i] = openvdb::FloatGrid::create(0);
            resGrid[i] = openvdb::FloatGrid::create(0);
            r2Grid[i] = openvdb::FloatGrid::create(0);
            pGrid[i] = openvdb::FloatGrid::create(0);
            ApGrid[i] = openvdb::FloatGrid::create(0);
        }
    }
    void mgSmokeData::resize(int levelNum)
    {
        tag = openvdb::Int32Grid::create(0);
        temperatureField.resize(levelNum);
        volumeField.resize(levelNum);
        pressField.resize(levelNum);
        iterBuffer.resize(levelNum);
        dx.resize(levelNum);
        for(int i=0;i<3;++i)
            velField[i].resize(levelNum);

        for(int i=0;i<levelNum;++i)
        {
            temperatureField[i] = openvdb::FloatGrid::create(0);
            volumeField[i] = openvdb::FloatGrid::create(0);
            pressField[i] = openvdb::FloatGrid::create(0);
            
            for(int xyz=0;xyz<3;++xyz)
                velField[xyz][i] = openvdb::FloatGrid::create(0);
        }

    }
    void mgSmokeData::initData(openvdb::FloatGrid::Ptr sdf, int levelNum, float inputdt)
    {
        dt = inputdt;
        dens = 1;
        resize(levelNum);
        printf("mgSmoke Data resize over!level Num is %d\n", levelNum);
        
        dx[0] = sdf->voxelSize()[0];
        for(int level = 1;level < levelNum;++level)
            dx[level] = dx[level-1]*2;

        auto tagtrans = openvdb::math::Transform::createLinearTransform(dx[levelNum-1]);
        tagtrans->postTranslate(openvdb::Vec3d{0.5,0.5,0.5}*dx[levelNum-1]);
        tag->setTransform(tagtrans);

        auto inputbbox = openvdb::CoordBBox();
        auto is_valid = sdf->tree().evalLeafBoundingBox(inputbbox);
        if (!is_valid) {
            return;
        }
        worldinputbbox = openvdb::BBoxd(sdf->indexToWorld(inputbbox.min()),
                                       sdf->indexToWorld(inputbbox.max()));
        worldinputbbox.min() += openvdb::Vec3d(-0.2);
        worldinputbbox.max() += openvdb::Vec3d(0.2);
        printf("generate world input box over!\n");

        openvdb::Vec3R loopbegin, loopend;
        tbb::blocked_range3d<int, int, int> pispace_range(0,0,0,0,0,0);

        //openvdb::tools::resampleToMatch<openvdb::tools::QuadraticSampler>(*sdf, *volumeField[level]);
        volumeField[0] = sdf->deepCopy();
        openvdb::tools::sdfToFogVolume(*volumeField[0]);
        temperatureField[0] = volumeField[0]->deepCopy();
        for(int level = 0;level < levelNum;++level){
            auto transform = openvdb::math::Transform::createLinearTransform(dx[level]);
            transform->postTranslate(openvdb::Vec3d{0.5,0.5,0.5}*dx[level]);
            temperatureField[level]->setTransform(transform);
            volumeField[level]->setTransform(transform);

            loopbegin =
                openvdb::tools::local_util::floorVec3(volumeField[level]->worldToIndex(worldinputbbox.min()));
            loopend =
                openvdb::tools::local_util::ceilVec3(volumeField[level]->worldToIndex(worldinputbbox.max()));
            pispace_range = tbb::blocked_range3d<int, int, int>(
                loopbegin.x(), loopend.x(), loopbegin.y(), loopend.y(), loopbegin.z(),
                loopend.z());
            {
                auto sdf_axr{sdf->getConstAccessor()};
                auto tag_axr{tag->getAccessor()};
                ConstBoxSample sdfSample(sdf_axr, sdf->transform());
                auto temperature_axr{temperatureField[level]->getAccessor()};
                auto volume_axr{volumeField[level]->getAccessor()};
                for (int i = pispace_range.pages().begin(); i < pispace_range.pages().end(); i += 1) {
                for (int j = pispace_range.rows().begin(); j < pispace_range.rows().end(); j += 1) {
                    for (int k = pispace_range.cols().begin(); k < pispace_range.cols().end(); k += 1) {
                        openvdb::Coord coord(i,j,k);
                        if(!temperature_axr.isValueOn(coord))
                        {
                            temperature_axr.setValue(coord, 0);
                            volume_axr.setValue(coord, 0);
                        }
                    }
                }
                }
            }
            //printf("start to mark active point leaves on level %d, levelNum is %d\n", level, levelNum);
            //tbb::parallel_for(pispace_range, mark_active_point_leaves);
            //tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), makeSubd);
            //printf("mark active point leaves on level %d, levelNum is %d end!!\n", level, levelNum);
            
            pressField[level]->setTree((std::make_shared<openvdb::FloatTree>(
                        temperatureField[level]->tree(), /*bgval*/ float(0),
                        openvdb::TopologyCopy())));
            for(int i=0;i<3;++i){
                auto velTrans = transform->copy();
                openvdb::Vec3d v(0);
                v[i] = -0.5 * double(dx[level]);
                velTrans->postTranslate(v);
                velField[i][level]->setTree((std::make_shared<openvdb::FloatTree>(
                            temperatureField[level]->tree(), /*bgval*/ float(0),
                            openvdb::TopologyCopy())));
                velField[i][level]->setTransform(velTrans);
                for (openvdb::FloatGrid::ValueOffIter iter = velField[i][level]->beginValueOff(); iter.test(); ++iter) {
                    iter.setValue(0);
                }
            }
            auto pressTrans = transform->copy();
            pressTrans->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(dx[level]));
            pressField[level]->setTransform(pressTrans);

            for (openvdb::FloatGrid::ValueOffIter iter = volumeField[level]->beginValueOff(); iter.test(); ++iter) {
                iter.setValue(0);
            }
            for (openvdb::FloatGrid::ValueOffIter iter = temperatureField[level]->beginValueOff(); iter.test(); ++iter) {
                iter.setValue(0);
            }
            for (openvdb::FloatGrid::ValueOffIter iter = pressField[level]->beginValueOff(); iter.test(); ++iter) {
                iter.setValue(10000);
            }
        }
        iterBuffer.init(pressField);
    };

    // temporally using semi-la advection
    void mgSmokeData::advection()
    {
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        //for(int level = 0;level < volumeField.size();++level)
        int level = 0;
        int sign;
        {
            sign = 1;
            volumeField[level]->tree().getNodes(leaves);
            auto new_temField = temperatureField[level]->deepCopy();
            auto new_volField = volumeField[level]->deepCopy();
            // advect the vertex attribute, temperature and volume
            auto semiLangAdvection = [&](const tbb::blocked_range<size_t> &r) 
            {
                openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> vel_axr[3]=
                    {velField[0][level]->getConstAccessor(),
                    velField[1][level]->getConstAccessor(),
                    velField[2][level]->getConstAccessor()};
                
                auto tem_axr = temperatureField[level]->getConstAccessor();
                auto vol_axr = volumeField[level]->getConstAccessor();
                auto new_tem_axr{new_temField->getAccessor()};
                auto new_vol_axr{new_volField->getAccessor()};
                ConstBoxSample velSampler[3] = {
                    ConstBoxSample(vel_axr[0], velField[0][level]->transform()),
                    ConstBoxSample(vel_axr[1], velField[1][level]->transform()),
                    ConstBoxSample(vel_axr[2], velField[2][level]->transform())
                };
                ConstBoxSample volSample(vol_axr, volumeField[level]->transform());
                ConstBoxSample temSample(tem_axr, temperatureField[level]->transform());
                // leaf iter
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                        if(!vol_axr.isValueOn(coord))
                            continue;
                        auto oldvol = vol_axr.getValue(coord);
                        auto wpos = temperatureField[level]->indexToWorld(coord);

                        openvdb::Vec3f vel, midvel;
                        for(int i=0;i<3;++i)
                        {
                            vel[i] = velSampler[i].wsSample(wpos);
                        }
                        //printf("vel is (%f,%f,%f)\n", vel[0], vel[1], vel[2]);
                        auto midwpos = wpos - sign * vel * 0.5 * dt;
                        for(int i=0;i<3;++i)
                        {    
                            midvel[i]  = velSampler[i].wsSample(midwpos);
                        }
                        
                        auto pwpos = wpos - sign * midvel * dt;
                        auto volume = volSample.wsSample(pwpos);
                        auto tem = temSample.wsSample(pwpos);
                        if(volume < 0.01)
                            volume = 0;
                        if(tem < 0.01)
                            tem = 0;
                        new_tem_axr.setValue(coord, tem);
                        new_vol_axr.setValue(coord, volume);
                    }
                }
            };
        
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), semiLangAdvection);
            sign = -1;
            auto temBuffer = temperatureField[level]->deepCopy(); 
            temperatureField[level]->clear(); 
            temperatureField[level] = new_temField->deepCopy();
            auto volBuffer = volumeField[level]->deepCopy(); 
            volumeField[level]->clear(); 
            volumeField[level]=new_volField->deepCopy();
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), semiLangAdvection);
            
            auto computeNewField = [&](const tbb::blocked_range<size_t> &r){
                auto tem_axr = temBuffer->getConstAccessor();
                auto vol_axr = volBuffer->getConstAccessor();
                auto new_tem_axr{new_temField->getAccessor()};
                auto new_vol_axr{new_volField->getAccessor()};
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                        if(!vol_axr.isValueOn(coord))
                            continue;
                        float temV = 1.5 * tem_axr.getValue(coord) - 0.5 * new_tem_axr.getValue(coord);
                        new_tem_axr.setValue(coord, 
                            temV);
                        new_vol_axr.setValue(coord, 
                            1.5 * vol_axr.getValue(coord) - 0.5 * new_vol_axr.getValue(coord));
                        //printf("tem value is %f\n", temV);
                    }
                }
            };
            
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeNewField);
            sign = 1;
            volumeField[level]->clear();
            volumeField[level] = new_volField->deepCopy();
            temperatureField[level]->clear();
            temperatureField[level] = new_temField->deepCopy();
            
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), semiLangAdvection);
            volumeField[level]->clear();temperatureField[level]->clear();
            volumeField[level] = new_volField->deepCopy();
            temperatureField[level] = new_temField->deepCopy();
            new_volField->clear();
            new_temField->clear();
            leaves.clear();

            velField[0][level]->tree().getNodes(leaves);
            openvdb::FloatGrid::Ptr new_vel[3], inte_vel[3];
            for(int i=0;i<3;++i)
            {
                new_vel[i] = velField[i][level]->deepCopy();
                inte_vel[i] = velField[i][level]->deepCopy();
            }
            
            auto velAdvection = [&](const tbb::blocked_range<size_t> &r){
                openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> 
                    vel_axr[3]=
                    {velField[0][level]->getConstAccessor(),
                    velField[1][level]->getConstAccessor(),
                    velField[2][level]->getConstAccessor()};
                openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> 
                    inte_axr[3]=
                    {inte_vel[0]->getConstAccessor(),inte_vel[1]->getConstAccessor(),inte_vel[2]->getConstAccessor()};
                openvdb::tree::ValueAccessor<openvdb::FloatTree, true> 
                    new_vel_axr[3]=
                    {new_vel[0]->getAccessor(),new_vel[1]->getAccessor(),new_vel[2]->getAccessor()};
                ConstBoxSample velSampler[3] = {
                    ConstBoxSample(inte_axr[0], inte_vel[0]->transform()),
                    ConstBoxSample(inte_axr[1], inte_vel[1]->transform()),
                    ConstBoxSample(inte_axr[2], inte_vel[2]->transform())
                };
                // leaf iter
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                        if(!vel_axr[0].isValueOn(coord))
                            continue;
                        // advect u,v,w separately
                        for(int i=0;i<3;++i)
                        {
                            auto wpos = velField[i][level]->indexToWorld(coord);
                            openvdb::Vec3f vel, midvel;
                            for(int j=0;j<3;++j)
                                vel[j] = velSampler[j].wsSample(wpos);
                        
                            auto midwpos = wpos - sign *0.5 * dt * vel;
                            for(int j=0;j<3;++j)
                                midvel[j] = velSampler[j].wsSample(midwpos);
                            auto pwpos = wpos - sign * dt * midvel;
                            auto pvel = velSampler[i].wsSample(pwpos);
                            new_vel_axr[i].setValue(coord, pvel);
                        }
                    }
                }

            };
            sign = 1;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), velAdvection);

            sign = -1;
            for(int i=0;i<3;++i)
                inte_vel[i] = new_vel[i]->deepCopy();
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), velAdvection);

            auto computeNewVel = [&](const tbb::blocked_range<size_t> &r){
                openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> 
                    vel_axr[3]=
                    {velField[0][level]->getConstAccessor(),velField[1][level]->getConstAccessor(),velField[2][level]->getConstAccessor()};
                openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> 
                    inte_axr[3]=
                    {inte_vel[0]->getConstAccessor(),inte_vel[1]->getConstAccessor(),inte_vel[2]->getConstAccessor()};
                openvdb::tree::ValueAccessor<openvdb::FloatTree, true> 
                    new_vel_axr[3]=
                    {new_vel[0]->getAccessor(),new_vel[1]->getAccessor(),new_vel[2]->getAccessor()};
                // leaf iter
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                        for(int i=0;i<3;++i){
                        if(!vel_axr[i].isValueOn(coord))
                            continue;
                        new_vel_axr[i].setValue(coord, 1.5 *vel_axr[i].getValue(coord) - 0.5 * new_vel_axr[i].getValue(coord));
                        }
                    }
                }
            };   
        
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeNewVel);
            for(int i=0;i<3;++i)
            {
                inte_vel[i]->clear();
                inte_vel[i] = new_vel[i]->deepCopy();
            }

            sign = 1;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), velAdvection);
            for(int i=0;i<3;++i)
            {
                inte_vel[i]->clear();velField[i][level]->clear();
                velField[i][level] = new_vel[i]->deepCopy();
                new_vel[i]->clear();
            }
            leaves.clear();
        }
    }

    void mgSmokeData::applyOuterforce(){
        float alpha =-0.1, beta = 0.2;
        int levelNum = dx.size();
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        for(int level = 0;level < levelNum; ++ level){
            auto applyOuterForce = [&](const tbb::blocked_range<size_t> &r) {
                openvdb::tree::ValueAccessor<openvdb::FloatTree, true> 
                    vel_axr[3]=
                    {velField[0][level]->getAccessor(),velField[1][level]->getAccessor(),velField[2][level]->getAccessor()};
                auto vol_axr = volumeField[level]->getConstAccessor();
                auto tem_axr = temperatureField[level]->getConstAccessor();
                ConstBoxSample volSample(vol_axr, volumeField[level]->transform());
                ConstBoxSample temSample(tem_axr, temperatureField[level]->transform());
                
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                        if(!vel_axr[1].isValueOn(coord))
                            continue;
                        auto wpos = velField[1][level]->indexToWorld(coord);
                        float volume =  volSample.wsSample(wpos);
                        if(volume <= 0)
                        {
                            vel_axr[1].setValue(coord, 0);
                            continue;
                        }
                        float temperature = temSample.wsSample(wpos);
                        float dens = (alpha * volume -beta * temperature);

                        auto vel = vel_axr[1].getValue(coord);
                        auto deltaV = vel - dens * 9.8 * dt;
                        // if(deltaV <= 0 && dens < 0.00001)
                        //     printf("deltaV is %f, dens is %f, vel is %f\n",
                        //     deltaV, dens, vel);
                        vel_axr[1].setValue(coord, deltaV);
                        
                    }
                }
            };

            velField[1][level]->tree().getNodes(leaves);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), applyOuterForce);
            leaves.clear();
        }
    }
    
    void mgSmokeData::Smooth(int level)
    {
        // using jacobi
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        float d2 = 1 / (dx[level] * dx[level]);
        auto delta = pressField[level]->deepCopy();
        auto computeDelta = [&](const tbb::blocked_range<size_t> &r){
            auto press_axr = pressField[level]->getConstAccessor();
            auto rhs_axr = iterBuffer.rhsGrid[level]->getConstAccessor();
            auto delta_axr = delta->getAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    if(!press_axr.isValueOn(coord))
                        continue;
                    float ipress = press_axr.getValue(coord);
                    float diag = 0, lapP = 0, rhs = rhs_axr.getValue(coord);
                    for(int ss = 0;ss<3;++ss)
                    for(int i = -1;i<=1;i+=2)
                    {
                        auto jcoord = coord;
                        jcoord[ss] += i;
                        if(!press_axr.isValueOn(jcoord))
                            continue;
                        float jpress = press_axr.getValue(jcoord);
                        lapP += (jpress - ipress) * d2;
                        diag -= d2;
                    }
                    if(diag != 0)
                    {
                        auto value = (rhs - lapP) / diag;
                        delta_axr.setValue(coord, value);
                    }
                    else
                    {
                        delta_axr.setValue(coord, 0);
                    }
                }
            }
        };
    
        auto applyDelta = [&](const tbb::blocked_range<size_t> &r){
            auto press_axr = pressField[level]->getAccessor();
            auto delta_axr = delta->getConstAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    if(!press_axr.isValueOn(coord))
                        continue;
                    float ipress = press_axr.getValue(coord);
                    ipress += 0.6667 * delta_axr.getValue(coord);
                    press_axr.setValue(coord, ipress);
                }
            }
        };

        pressField[level]->tree().getNodes(leaves);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeDelta);
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), applyDelta);
        
        leaves.clear();
    }

    void mgSmokeData::PossionSolver()
    {
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        int level = dx.size()-1;
        float dens = 1, alpha, beta;
        auto initIter = [&](const tbb::blocked_range<size_t> &r) {
            auto press_axr = pressField[level]->getConstAccessor();
            auto rhs_axr = iterBuffer.rhsGrid[level]->getAccessor();
            auto res_axr = iterBuffer.resGrid[level]->getAccessor();
            auto p_axr = iterBuffer.pGrid[level]->getAccessor();
            auto r2_axr = iterBuffer.r2Grid[level]->getAccessor();
            openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> 
                vel_axr[3]=
                {velField[0][level]->getConstAccessor(),velField[1][level]->getConstAccessor(),velField[2][level]->getConstAccessor()};
            
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!press_axr.isValueOn(coord))
                        continue;
                    auto rhs = rhs_axr.getValue(coord);
                    // compute Ax
                    float Ax = 0;
                    for(int i=-1;i<=1;i+=2)
                    for(int ss =0;ss<3;++ss)
                    {
                        auto ipos =coord;
                        ipos[ss] += i;
                        Ax += press_axr.getValue(ipos);
                    }
                    Ax = (6 * press_axr.getValue(coord) - Ax) / (dx[level] * dx[level]);
                    Ax *= dt / dens;

                    res_axr.setValue(coord, rhs - Ax);
                    p_axr.setValue(coord, rhs - Ax);
                    r2_axr.setValue(coord, rhs - Ax);
                }
            }
        };
    
        auto reductionR2 = [&](const tbb::blocked_range<size_t> &r, float r2Sum){
            auto r2_axr = iterBuffer.r2Grid[level]->getAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!r2_axr.isValueOn(coord))
                        continue;
                    float r2 = r2_axr.getValue(coord);
                    r2Sum += r2 * r2;
                }
            }
            return r2Sum;
        };

        auto reductionR = [&](const tbb::blocked_range<size_t> &r, float rSum){
            auto res_axr = iterBuffer.resGrid[level]->getAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!res_axr.isValueOn(coord))
                        continue;
                    float r2 = res_axr.getValue(coord);
                    rSum += r2 * r2;
                }
            }
            return rSum;
        };
        
        auto reductionPAP = [&](const tbb::blocked_range<size_t> &r, float pApSum){
            auto Ap_axr = iterBuffer.ApGrid[level]->getAccessor();
            auto p_axr = iterBuffer.pGrid[level]->getAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!Ap_axr.isValueOn(coord))
                        continue;
                    float Ap = Ap_axr.getValue(coord);
                    pApSum += Ap * p_axr.getValue(coord);
                }
            }
            return pApSum;
        };

        auto computeAp = [&](const tbb::blocked_range<size_t> &r){
            auto p_axr = iterBuffer.pGrid[level]->getConstAccessor();
            auto Ap_axr = iterBuffer.ApGrid[level]->getAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!p_axr.isValueOn(coord))
                        continue;
                    // compute Ap
                    float Ap = 0;
                    for(int i=-1;i<=1;i+=2)
                    for(int ss =0;ss<3;++ss)
                    {
                        auto ipos =coord;
                        ipos[ss] += i;
                        Ap += p_axr.getValue(ipos);
                    }
                    Ap = (6 * p_axr.getValue(coord) - Ap) / (dx[level] * dx[level]);
                    Ap *= dt / dens;

                    Ap_axr.setValue(coord, Ap);
                }
            }
        };
        
        auto updatePress = [&](const tbb::blocked_range<size_t> &r){
            auto press_axr = pressField[level]->getAccessor();
            auto res_axr = iterBuffer.resGrid[level]->getConstAccessor();
            auto Ap_axr = iterBuffer.ApGrid[level]->getConstAccessor();
            auto p_axr = iterBuffer.pGrid[level]->getConstAccessor();
            auto r2_axr = iterBuffer.r2Grid[level]->getAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!press_axr.isValueOn(coord))
                        continue;
                    float press = press_axr.getValue(coord);
                    press_axr.setValue(coord, press + alpha * p_axr.getValue(coord));
                    float r = res_axr.getValue(coord);
                    float ap = Ap_axr.getValue(coord);
                    r2_axr.setValue(coord, r - alpha * ap);
                }
            }
        };
        
        auto updateP = [&](const tbb::blocked_range<size_t> &r){
            auto res_axr = iterBuffer.resGrid[level]->getAccessor();
            auto p_axr = iterBuffer.pGrid[level]->getAccessor();
            auto r2_axr = iterBuffer.r2Grid[level]->getConstAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!r2_axr.isValueOn(coord))
                        continue;
                    float r2 = r2_axr.getValue(coord);
                    float p = r2 + beta * p_axr.getValue(coord);
                    p_axr.setValue(coord, p);
                    res_axr.setValue(coord, r2);
                }
            }
        };

        pressField[level]->tree().getNodes(leaves);
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), initIter);
        int voxelNum = pressField[level]->activeVoxelCount();
        alpha = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leaves.size()), 0.0f,reductionR2, std::plus<float>());
        if(alpha < 0.01 * voxelNum)
        {
            printf("alpha is %f, voxel num is %d\n", alpha, voxelNum);
            return;
        }
        printf("start to iter, alpha is %f\n", alpha);
        for(int k=0;k < 100;++k)
        {
            alpha = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leaves.size()), 0.0f,reductionR, std::plus<float>());
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeAp);
            beta = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leaves.size()), 0.0f,reductionPAP, std::plus<float>());
            alpha /= beta;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), updatePress);
            beta = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leaves.size()), 0.0f,reductionR2, std::plus<float>());
            if(beta < 0.00001 * voxelNum)
                break;
            printf("new beta is %f\n", beta);
            alpha = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leaves.size()), 0.0f,reductionR, std::plus<float>());
            beta /= alpha;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), updateP);
            
        }
        printf("iter over. error is %f\n", beta);
        leaves.clear();
        
    }
    void mgSmokeData::Restrict(int level)
    {
        int coarseLevel = level + 1;
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        iterBuffer.resGrid[coarseLevel]->tree().getNodes(leaves);
        auto computeCoarse = [&](const tbb::blocked_range<size_t> &r) {
            auto press_axr = pressField[coarseLevel]->getAccessor();
            auto res_axr = iterBuffer.resGrid[level]->getConstAccessor();
            ConstBoxSample resSampler(res_axr, iterBuffer.resGrid[level]->transform());
            auto rhs_axr = iterBuffer.rhsGrid[coarseLevel]->getAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    if(!press_axr.isValueOn(coord))
                        continue;
                    auto wpos = iterBuffer.resGrid[coarseLevel]->indexToWorld(coord);
                    auto resValue = resSampler.wsSample(wpos);
                    rhs_axr.setValue(coord, resValue);
                    press_axr.setValue(coord, 0);
                }
            }
        };

        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeCoarse);
    
    }

    void mgSmokeData::Prolongate(int level){
        int coarseLevel = level + 1;
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        pressField[coarseLevel]->tree().getNodes(leaves);
        auto computeFine = [&](const tbb::blocked_range<size_t> &r) {
            auto press_axr = pressField[coarseLevel]->getConstAccessor();
            ConstBoxSample pressSampler(press_axr, pressField[coarseLevel]->transform());
            auto fine_press_axr = pressField[level]->getAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    if(!fine_press_axr.isValueOn(coord))
                        continue;
                    auto wpos = pressField[coarseLevel]->indexToWorld(coord);
                    auto pressValue = pressSampler.wsSample(wpos);
                    
                    fine_press_axr.setValue(coord, pressValue);
                }
            }
        };
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeFine);
            
    }
    void mgSmokeData::solvePress()
    {
        int levelNum = dx.size();
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;

        auto computeRHS = [&](const tbb::blocked_range<size_t> &r) {
            openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> 
                vel_axr[3] = 
                {velField[0][0]->getConstAccessor(),
                velField[1][0]->getConstAccessor(),
                velField[2][0]->getConstAccessor()};
            auto press_axr = pressField[0]->getConstAccessor();
            auto rhs_axr = iterBuffer.rhsGrid[0]->getAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    if(!press_axr.isValueOn(coord))
                        continue;
                    float rhs = 0;
                    for(int i = -1;i <= 0; ++i)
                    for(int ss = 0;ss < 3; ++ss)
                    {
                        auto ipos = coord;
                        ipos[ss] += i;
                        float vel = vel_axr[ss].getValue(ipos);
                        rhs -= (i+0.5)*2*vel/dx[0];
                    }
                    rhs_axr.setValue(coord, rhs);
                }
            }
        };
        pressField[0]->tree().getNodes(leaves);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeRHS);
        leaves.clear();
        for(int level = 0;level<levelNum-1;++level)
        {
            float d2 = 1 / (dx[level] * dx[level]);
            pressField[level]->tree().getNodes(leaves);
            
            Smooth(level);

            auto computeRes = [&](const tbb::blocked_range<size_t> &r){
                auto press_axr = pressField[level]->getConstAccessor();
                auto rhs_axr = iterBuffer.rhsGrid[level]->getConstAccessor();
                auto res_axr = iterBuffer.resGrid[level]->getAccessor();
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                        if(!press_axr.isValueOn(coord))
                            continue;
                        float ipress = press_axr.getValue(coord);
                        float diag = 0, lapP = 0, rhs = rhs_axr.getValue(coord);
                        for(int ss = 0;ss<3;++ss)
                        for(int i = -1;i<=1;i+=2)
                        {
                            auto jcoord = coord;
                            jcoord[ss] += i;
                            if(!press_axr.isValueOn(jcoord))
                                continue;
                            float jpress = press_axr.getValue(jcoord);
                            lapP += (jpress - ipress) * d2;
                            diag -= d2;
                        }
                        if(diag != 0)
                        {
                            auto value = (rhs - lapP) / diag;
                            res_axr.setValue(coord, value);
                        }
                        else
                        {
                            res_axr.setValue(coord, 0);
                        }
                    }
                }
            };
    
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeRes);
            
            //if(level < levelNum - 1)
            Restrict(level);

            leaves.clear();
        }
        // step2: solve coarset level grid
        PossionSolver();
        // pro longate
        for(int level = levelNum-2;level>=0;--level)
        {
            Prolongate(level);
            Smooth(level);
        }

        auto applyPress = [&](const tbb::blocked_range<size_t> &r){
            auto press_axr = pressField[0]->getAccessor();
            openvdb::tree::ValueAccessor<openvdb::FloatTree, true> 
                vel_axr[3]=
                {velField[0][0]->getAccessor(),velField[1][0]->getAccessor(),velField[2][0]->getAccessor()};
            
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    float gradPV[3], velV[3];
                    for(int i=0;i<3;++i)
                    {
                        if(!vel_axr[i].isValueOn(coord))
                            continue;
                        float vel = vel_axr[i].getValue(coord);
                        float gradP = 0;
                        for(int j=0;j<=1;++j){
                            auto ipos = coord;
                            ipos[i] += j;
                            gradP += (j-0.5)*2*press_axr.getValue(ipos);
                        }
                        velV[i] = vel;
                        vel -= dt * gradP /(dx[0] * dens);
                        float mol = abs(vel);
                        if(mol > 2)
                            vel = vel / mol * 2;
                        vel *= 0.99;
                        
                        gradPV[i] = gradP;
                        vel_axr[i].setValue(coord, vel);
                    }
                }
            }
        };
        
        leaves.clear();
        velField[0][0]->tree().getNodes(leaves);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), applyPress);

    }
    void mgSmokeData::step(){
        //printf("begin to step\n");
        advection();
        //printf("advection over\n");
        applyOuterforce();
        //printf("apply outer force over\n");
        solvePress();
        
    }

    struct mgSmokeToSDF : zeno::INode{
        virtual void apply() override {
            auto data = get_input<mgSmokeData>("mgSmokeData");
            std::cout<<"data pointer: "<<data<<" vole pointer: "<<data->volumeField[0]<<std::endl;
            
            auto result = zeno::IObject::make<VDBFloatGrid>();
            int levelNum = data->volumeField.size();
            //result->m_grid = openvdb::FloatGrid::create(0);
            // result->m_grid->setTransform(data->volumeField[levelNum-1]->transformPtr());
            //result->m_grid->setTransform(data->volumeField[0]->transformPtr());
            
            result->m_grid = openvdb::tools::fogToSdf(*(data->volumeField[0]), 0);
            // std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
            // for(int level = levelNum-1;level < levelNum;++level)
            // {
            //     float dx = data->dx[level];
            //     auto volumeField = data->volumeField[level];
            //     volumeField->tree().getNodes(leaves);

            //     auto sampleValue = [&](const tbb::blocked_range<size_t> &r){
            //         auto volume_axr = volumeField->getConstAccessor();
            //         auto result_axr = result->m_grid->getAccessor();
            //         for (auto liter = r.begin(); liter != r.end(); ++liter) {
            //             auto &leaf = *leaves[liter];
            //             for (auto offset = 0; offset < leaf.SIZE; ++offset) {
            //                 openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
            //                 if(!volume_axr.isValueOn(coord))
            //                     continue;
            //                 auto vol = volume_axr.getValue(coord);
            //                 if(vol <= 0.01)
            //                     continue;
            //                 auto wpos = volumeField->indexToWorld(coord);
            //                 auto fine = result->m_grid->worldToIndex(wpos);
            //                 auto finepos = openvdb::Coord(round(fine[0]), round(fine[1]), round(fine[2]));
            //                 result_axr.setValue(finepos, vol);
            //                 //printf("vol is %f, wpos is (%f,%f,%f),fineindex is (%d,%d,%d)\n",
            //                 //    vol, wpos[0],wpos[1],wpos[2], finepos[0],finepos[1],finepos[2]);
            //             }
            //         }
            //     };
                
            //     tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), sampleValue);
            //     leaves.clear();
            // }
            
            printf("mgsmoke to vdb is done. result nodes num is %d\n",
                result->m_grid->tree().activeVoxelCount());
            set_output("volumeGrid", result);
        }
    };
    ZENDEFNODE(mgSmokeToSDF,
    {
        {"mgSmokeData"},
        {"volumeGrid"},
        {},
        {"AdaptiveSolver"},
    }
    );

    struct SDFtoMGSmoke : zeno::INode{
        virtual void apply() override {
            auto sdf = get_input<zeno::VDBFloatGrid>("sdfgrid");
            auto data = zeno::IObject::make<mgSmokeData>();
            float dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
            int levelNum = get_input("levelNum")->as<zeno::NumericObject>()->get<int>();
            
            data->initData(sdf->m_grid, levelNum, dt);
            printf("sdf to mg smoke done\n");
            set_output("mgSmokeData", data);
        }
    };
    ZENDEFNODE(SDFtoMGSmoke,
    {
        {"sdfgrid", "dt", "levelNum"},
        {"mgSmokeData"},
        {},
        {"AdaptiveSolver"},
    }
    );

    struct mgSmokeSolver : zeno::INode{
        virtual void apply() override {
            auto data = get_input<mgSmokeData>("mgSmokeData");
            data->step();
            //printf("volume num is %d\n", data->volumeField->tree().activeVoxelCount());
            set_output("mgSmokeData", data);
        }
    };
    ZENDEFNODE(mgSmokeSolver,
    {
        {"mgSmokeData"},
        {"mgSmokeData"},
        {},
        {"AdaptiveSolver"},
    }
    );
}