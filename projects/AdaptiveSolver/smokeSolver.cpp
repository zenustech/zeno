#include "smokeSolver.h"
#include <openvdb/tools/Interpolation.h>
#include "tbb/blocked_range3d.h"
namespace zeno{
    void iterStuff::init(openvdb::FloatGrid::Ptr pressField)
    {
        rhsGrid = pressField->deepCopy();
        resGrid = pressField->deepCopy();
        r2Grid = pressField->deepCopy();
        pGrid = pressField->deepCopy();
        ApGrid = pressField->deepCopy();
    }

    void smokeData::initData(openvdb::FloatGrid::Ptr sdf, float inputdt)
    {
        dt = inputdt;
        dx = sdf->voxelSize()[0];
        temperatureField = openvdb::FloatGrid::create(0);
        volumeField = openvdb::FloatGrid::create(0);
        pressField = openvdb::FloatGrid::create(0);
        for(int i=0;i<3;++i)
            velField[i] = zeno::IObject::make<VDBFloatGrid>()->m_grid;
        // set up transform
        auto transform = sdf->transformPtr()->copy();
        temperatureField->setTransform(transform);
        volumeField->setTransform(transform);
        
        auto inputbbox = openvdb::CoordBBox();
        auto is_valid = sdf->tree().evalLeafBoundingBox(inputbbox);
        if (!is_valid) {
            return;
        }
        worldinputbbox = openvdb::BBoxd(sdf->indexToWorld(inputbbox.min()),
                                       sdf->indexToWorld(inputbbox.max()));
        worldinputbbox.min() += openvdb::Vec3d(-0.1);
        worldinputbbox.max() -= openvdb::Vec3d(-0.1);
        
        auto loopbegin =
            openvdb::tools::local_util::floorVec3(sdf->worldToIndex(worldinputbbox.min()));
        auto loopend =
            openvdb::tools::local_util::ceilVec3(sdf->worldToIndex(worldinputbbox.max()));

        auto pispace_range = tbb::blocked_range3d<int, int, int>(
            loopbegin.x(), loopend.x(), loopbegin.y(), loopend.y(), loopbegin.z(),
            loopend.z());
        auto mark_active_point_leaves =
            [&](const tbb::blocked_range3d<int, int, int> &r) {
                auto sdf_axr{sdf->getConstAccessor()};
                auto temperature_axr{temperatureField->getAccessor()};
                auto volume_axr{volumeField->getAccessor()};
                for (int i = r.pages().begin(); i < r.pages().end(); i += 1) {
                for (int j = r.rows().begin(); j < r.rows().end(); j += 1) {
                    for (int k = r.cols().begin(); k < r.cols().end(); k += 1) {
                        openvdb::Coord coord(i,j,k);
                        if(!sdf_axr.isValueOn(coord) || sdf_axr.getValue(coord) >= 0)
                        {
                            temperature_axr.setValue(coord, 0);
                            volume_axr.setValue(coord, 0);
                        }
                        else
                        {
                            temperature_axr.setValue(coord, 1);
                            volume_axr.setValue(coord, 1);
                        }
                    }     // loop k
                }       // loop j
                }         // loop i
            };          // end mark active point elaves
        tbb::parallel_for(pispace_range, mark_active_point_leaves);

        pressField->setTree((std::make_shared<openvdb::FloatTree>(
                    temperatureField->tree(), /*bgval*/ float(0),
                    openvdb::TopologyCopy())));
        for(int i=0;i<3;++i){
            auto velTrans = transform->copy();
            openvdb::Vec3d v(0);
            v[i] = -0.5 * double(dx);
            velTrans->postTranslate(v);
            velField[i]->setTree((std::make_shared<openvdb::FloatTree>(
                        temperatureField->tree(), /*bgval*/ float(0),
                        openvdb::TopologyCopy())));
            velField[i]->setTransform(velTrans);
            for (openvdb::FloatGrid::ValueOffIter iter = velField[i]->beginValueOff(); iter.test(); ++iter) {
                iter.setValue(0);
            }
        }
        auto pressTrans = transform->copy();
        pressTrans->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(dx));
        pressField->setTransform(pressTrans);

        for (openvdb::FloatGrid::ValueOffIter iter = volumeField->beginValueOff(); iter.test(); ++iter) {
            iter.setValue(0);
        }
        for (openvdb::FloatGrid::ValueOffIter iter = temperatureField->beginValueOff(); iter.test(); ++iter) {
            iter.setValue(0);
        }
        for (openvdb::FloatGrid::ValueOffIter iter = pressField->beginValueOff(); iter.test(); ++iter) {
            iter.setValue(0);
        }
        iterBuffer.init(pressField);
    }

    void smokeData::advection()
    {
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        auto new_temField = temperatureField->deepCopy();
        auto new_volField = volumeField->deepCopy();
        int sign = 1;
        
        auto semiLangAdvection = [&](const tbb::blocked_range<size_t> &r) {
            openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> vel_axr[3]=
                {velField[0]->getConstAccessor(),velField[1]->getConstAccessor(),velField[2]->getConstAccessor()};
            
            auto tem_axr = temperatureField->getConstAccessor();
            auto vol_axr = volumeField->getConstAccessor();
            auto new_tem_axr{new_temField->getAccessor()};
            auto new_vol_axr{new_volField->getAccessor()};
            ConstBoxSample velSampler[3] = {
                ConstBoxSample(vel_axr[0], velField[0]->transform()),
                ConstBoxSample(vel_axr[1], velField[1]->transform()),
                ConstBoxSample(vel_axr[2], velField[2]->transform())
            };
            ConstBoxSample volSample(vol_axr, volumeField->transform());
            ConstBoxSample temSample(tem_axr, temperatureField->transform());
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    if(!vol_axr.isValueOn(coord))
                        continue;
                    auto wpos = temperatureField->indexToWorld(coord);

                    openvdb::Vec3f vel, midvel;
                    for(int i=0;i<3;++i)
                    {
                        vel[i] = velSampler[i].wsSample(wpos);
                    }
                    
                    auto midwpos = wpos - sign * vel * 0.5 * dt;
                    for(int i=0;i<3;++i)
                    {    
                        midvel[i]  = velSampler[i].wsSample(midwpos);
                    }
                    
                    auto pwpos = wpos - sign * midvel * dt;
                    auto volume = volSample.wsSample(pwpos);
                    auto tem = temSample.wsSample(pwpos);
                    // if(abs(volume) > 1 || abs(tem) > 1)
                    //     printf("vol is %f, tem is %f, pwpos is (%f,%f,%f), wpos is (%f,%f,%f)\n", 
                    //         volume, tem, pwpos[0],pwpos[1],pwpos[2],
                    //         wpos[0],wpos[1],wpos[2]);
                    new_tem_axr.setValue(coord, tem);

                    new_vol_axr.setValue(coord, volume);
                }
            }
        };

        volumeField->tree().getNodes(leaves);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), semiLangAdvection);
        sign = -1;
        auto temBuffer = temperatureField->deepCopy(); 
        temperatureField->clear(); 
        temperatureField = new_temField->deepCopy();
        auto volBuffer = volumeField->deepCopy(); 
        volumeField->clear(); 
        volumeField=new_volField->deepCopy();
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
        volumeField->clear();
        volumeField = new_volField->deepCopy();
        temperatureField->clear();
        temperatureField = new_temField->deepCopy();
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), semiLangAdvection);
        volumeField->clear();temperatureField->clear();
        volumeField = new_volField->deepCopy();
        temperatureField = new_temField->deepCopy();
        
        leaves.clear();
        openvdb::FloatGrid::Ptr new_vel[3], inte_vel[3];
        for(int i=0;i<3;++i)
        {
            new_vel[i] = velField[i]->deepCopy();
            inte_vel[i] = velField[i];
        }
        sign = 1;
        auto velAdvection = [&](const tbb::blocked_range<size_t> &r){
            openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> 
                vel_axr[3]=
                {velField[0]->getConstAccessor(),velField[1]->getConstAccessor(),velField[2]->getConstAccessor()};
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
                        auto wpos = velField[i]->indexToWorld(coord);
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
        sign = -1;
        for(int i=0;i<3;++i)
            inte_vel[i] = new_vel[i]->deepCopy();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), velAdvection);
        
        auto computeNewVel = [&](const tbb::blocked_range<size_t> &r){
            openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> 
                vel_axr[3]=
                {velField[0]->getConstAccessor(),velField[1]->getConstAccessor(),velField[2]->getConstAccessor()};
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
            inte_vel[i]->clear();velField[i]->clear();
            velField[i] = new_vel[i]->deepCopy();
        }

    }

    void smokeData::applyOuterforce()
    {
        float alpha =-0.1, beta = 0.2;
        
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        auto applyOuterForce = [&](const tbb::blocked_range<size_t> &r) {
            openvdb::tree::ValueAccessor<openvdb::FloatTree, true> 
                vel_axr[3]=
                {velField[0]->getAccessor(),velField[1]->getAccessor(),velField[2]->getAccessor()};
            // BoxSample velSample[3] = {
            //     BoxSample(vel_axr[0], velField[0]->transform()),
            //     BoxSample(vel_axr[1], velField[1]->transform()),
            //     BoxSample(vel_axr[2], velField[2]->transform())
            // };
            auto vol_axr = volumeField->getConstAccessor();
            auto tem_axr = temperatureField->getConstAccessor();
            ConstBoxSample volSample(vol_axr, volumeField->transform());
            ConstBoxSample temSample(tem_axr, temperatureField->transform());
            
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    //for(int i=0;i<3;++i)
                    int i = 1;
                    {
                        if(!vel_axr[i].isValueOn(coord))
                            continue;
                        auto wpos = velField[i]->indexToWorld(coord);
                        float volume =  volSample.wsSample(wpos);
                        float temperature = temSample.wsSample(wpos);
                        float dens = (alpha * volume -beta * temperature);

                        auto vel = vel_axr[i].getValue(coord);
                        auto deltaV = vel - dens * 9.8 * dt;
                        //if(abs(deltaV) > 0.1)
                        //    printf("%d's vel is %f, vol is %f, tem is %f, dens is %f, wpos is (%f,%f,%f)\n", 
                        //        i, deltaV, volume, temperature, dens, wpos[0], wpos[1], wpos[2]);
                        vel_axr[i].setValue(coord, deltaV);
                    }
                }
            }
        };

        velField[0]->tree().getNodes(leaves);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), applyOuterForce);
        
    }
    
    void smokeData::solvePress(){
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        float dens = 1, alpha, beta;
        auto initIter = [&](const tbb::blocked_range<size_t> &r) {
            auto press_axr = pressField->getConstAccessor();
            auto rhs_axr = iterBuffer.rhsGrid->getAccessor();
            auto res_axr = iterBuffer.resGrid->getAccessor();
            auto p_axr = iterBuffer.pGrid->getAccessor();
            auto r2_axr = iterBuffer.r2Grid->getAccessor();
            openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> 
                vel_axr[3]=
                {velField[0]->getConstAccessor(),velField[1]->getConstAccessor(),velField[2]->getConstAccessor()};
            
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!press_axr.isValueOn(coord))
                        continue;
                    // compute b
                    float rhs = 0;
                    for(int i = -1;i <= 0; ++i)
                    for(int ss = 0;ss < 3; ++ss)
                    {
                        auto ipos = coord;
                        ipos[ss] += i;
                        float vel = vel_axr[ss].getValue(ipos);
                        rhs -= (i+0.5)*2*vel/dx;
                    }
                    rhs_axr.setValue(coord, rhs);
                    // compute Ax
                    float Ax = 0;
                    for(int i=-1;i<=1;i+=2)
                    for(int ss =0;ss<3;++ss)
                    {
                        auto ipos =coord;
                        ipos[ss] += i;
                        Ax += press_axr.getValue(ipos);
                    }
                    Ax = (6 * press_axr.getValue(coord) - Ax) / (dx * dx);
                    Ax *= dt / dens;

                    res_axr.setValue(coord, rhs - Ax);
                    p_axr.setValue(coord, rhs - Ax);
                    r2_axr.setValue(coord, rhs - Ax);
                }
            }
        };
    
        auto reductionR2 = [&](const tbb::blocked_range<size_t> &r, float r2Sum){
            auto r2_axr = iterBuffer.r2Grid->getAccessor();
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
            auto res_axr = iterBuffer.resGrid->getAccessor();
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
            auto Ap_axr = iterBuffer.ApGrid->getAccessor();
            auto p_axr = iterBuffer.pGrid->getAccessor();
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
            auto p_axr = iterBuffer.pGrid->getConstAccessor();
            auto Ap_axr = iterBuffer.ApGrid->getAccessor();
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
                    Ap = (6 * p_axr.getValue(coord) - Ap) / (dx * dx);
                    Ap *= dt / dens;

                    Ap_axr.setValue(coord, Ap);
                }
            }
        };
        
        auto updatePress = [&](const tbb::blocked_range<size_t> &r){
            auto press_axr = pressField->getAccessor();
            auto res_axr = iterBuffer.resGrid->getConstAccessor();
            auto Ap_axr = iterBuffer.ApGrid->getConstAccessor();
            auto p_axr = iterBuffer.pGrid->getConstAccessor();
            auto r2_axr = iterBuffer.r2Grid->getAccessor();
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
            auto res_axr = iterBuffer.resGrid->getAccessor();
            auto p_axr = iterBuffer.pGrid->getAccessor();
            auto r2_axr = iterBuffer.r2Grid->getConstAccessor();
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

        pressField->tree().getNodes(leaves);
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), initIter);
        int voxelNum = pressField->activeVoxelCount();
        alpha = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leaves.size()), 0.0f,reductionR2, std::plus<float>());
        if(alpha < 0.01 * voxelNum)
            return;
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

        auto applyPress = [&](const tbb::blocked_range<size_t> &r){
            auto press_axr = pressField->getAccessor();
            openvdb::tree::ValueAccessor<openvdb::FloatTree, true> 
                vel_axr[3]=
                {velField[0]->getAccessor(),velField[1]->getAccessor(),velField[2]->getAccessor()};
            
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
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
                        vel -= dt * gradP /(dx * dens);
                        float mol = abs(vel);
                        if(mol > 2)
                            vel = vel / mol * 2;
                        vel *= 0.99;
                        vel_axr[i].setValue(coord, vel);
                        
                    }
                }
            }
        };
        leaves.clear();
        velField[0]->tree().getNodes(leaves);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), applyPress);
        
    }
    void smokeData::step(){
        printf("begin to step\n");
        advection();
        printf("advection over\n");
        applyOuterforce();
        printf("apply outer force over\n");
        solvePress();
    }
    struct SDFtoSmoke : zeno::INode{
        virtual void apply() override {
            auto sdf = get_input<zeno::VDBFloatGrid>("sdfgrid");
            auto data = zeno::IObject::make<smokeData>();
            float dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
            data->initData(sdf->m_grid, dt);
            printf("after init, volume num is %d\n", data->volumeField->tree().activeVoxelCount());
            
            set_output("smokeData", data);
        }
    };
    ZENDEFNODE(SDFtoSmoke,
    {
        {"sdfgrid", "dt"},
        {"smokeData"},
        {},
        {"AdaptiveSolver"},
    }
    );

    struct smokeSolver : zeno::INode{
        virtual void apply() override {
            auto data = get_input<smokeData>("smokeData");
            data->step();
            printf("volume num is %d\n", data->volumeField->tree().activeVoxelCount());
            set_output("smokeData", data);
        }
    };
    ZENDEFNODE(smokeSolver,
    {
        {"smokeData"},
        {"smokeData"},
        {},
        {"AdaptiveSolver"},
    }
    );

    struct smokeToSDF : zeno::INode{
        virtual void apply() override {
            auto data = get_input<smokeData>("smokeData");
            std::cout<<"data pointer: "<<data<<std::endl;
            //printf()
            printf("smoke sdf:volume num is %d\n", data->volumeField->tree().activeVoxelCount());
            
            auto result = zeno::IObject::make<VDBFloatGrid>();
            result->m_grid = data->volumeField;
            printf("heiha ,after set result !\n");
            set_output("volumeGrid", result);
        }
    };
    ZENDEFNODE(smokeToSDF,
    {
        {"smokeData"},
        {"volumeGrid"},
        {},
        {"AdaptiveSolver"},
    }
    );

}