#include "AdaptiveSolver.h"
#include <openvdb/Types.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
namespace zeno{

struct AdaptiveSolver : zeno::INode
{
    std::shared_ptr<mgData>  data;
    int frameNum;
    float dt, density, dx;
    float alpha;
    float beta;
    openvdb::FloatGrid::Ptr sdfgrid;
    openvdb::Vec3fGrid::Ptr velGrid;
    openvdb::FloatGrid::Ptr pressGrid;
    openvdb::FloatGrid::Ptr staggeredSDFGrid;
    
    // iteration terms
    openvdb::FloatGrid::Ptr rhsGrid;
    openvdb::FloatGrid::Ptr resGrid;
    openvdb::FloatGrid::Ptr r2Grid;
        
    openvdb::FloatGrid::Ptr pGrid;
    openvdb::FloatGrid::Ptr ApGrid;

    virtual void apply() override {
        data = get_input<mgData>("mgData");
        dt = has_input("dt") ? get_input("dt")->as<NumericObject>()->get<float>()
            :0.001;
        density = has_input("density") ? get_input("density")->as<NumericObject>()->get<float>()
            :1000.0f;
        frameNum = has_input("frameNum") ? get_input("frameNum")->as<NumericObject>()->get<int>()
            :0;
        if(frameNum % 10 == 0)
            data->recomputeSDF();
        step();
        //generate  
        set_output("mgData", data);
    }
    
    //using cg iteration to solve press possion equation 
    void step()
    {
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        velGrid = data->vel[0];
        pressGrid = data->press[0];
        staggeredSDFGrid = data->staggeredSDF[0];
        // iteration terms
        rhsGrid = data->rhs[0];
        resGrid = data->residual[0];
        r2Grid = data->r2[0];
        
        pGrid = data->p[0];
        ApGrid = data->Ap[0];

        dx = data->hLevels[0];
        // compute the finest level only
        sdfgrid = data->sdf[0];
        
        openvdb::Int32Grid::Ptr tag = zeno::IObject::make<VDBIntGrid>()->m_grid;
        tag->setTree(std::make_shared<openvdb::FloatTree>(
            sdfgrid->tree(), /*bgval*/ float(1),
            openvdb::TopologyCopy()));

        auto applyGravityAndBound = [&](const tbb::blocked_range<size_t> &r) {
            auto vel_axr = velGrid->getAccessor();
            auto sdf_axr = sdfgrid->getConstAccessor();
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelipos = leaf.offsetToGlobalCoord(offset);

                    if(!sdf_axr.isValueOn(openvdb::Coord(voxelipos)))
                        continue;
                    if(sdf_axr.getValue(voxelipos) > 0)
                    {
                        vel_axr.setValue(voxelipos, openvdb::Vec3f(0));
                        continue;
                    }
                    openvdb::Vec3f vel_value = vel_axr.getValue(openvdb::Coord(voxelipos));
                    vel_value += openvdb::Vec3f(0, -dt * 9.8, 0);
                    // bound
                    // if(voxelipos[1] <= -20)
                    //     vel_value = openvdb::Vec3f(vel_value[0], 0.0f, vel_value[2]);
                    // if(voxelipos[0] <= -30 || voxelipos[0] >= 30)
                    //     vel_value = openvdb::Vec3f(0.0f, vel_value[1], vel_value[2]);
                    // if(voxelipos[2] <= -30 || voxelipos[2] >= 30)
                    //     vel_value = openvdb::Vec3f(vel_value[0], vel_value[1], 0.0f);
                    
                    vel_axr.setValue(openvdb::Coord(voxelipos), vel_value);
                }
            }
        };
        
        // set r0 and p0
        auto initIter = [&](const tbb::blocked_range<size_t> &r) {
            auto press_axr{pressGrid->getAccessor()};
            auto rhs_axr{rhsGrid->getAccessor()};
            auto res_axr{resGrid->getAccessor()};
            auto p_axr{pGrid->getAccessor()};
            auto sdf_axr{sdfgrid->getConstAccessor()};
            auto ssdf_axr{staggeredSDFGrid->getConstAccessor()};
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    
                    if(!press_axr.isValueOn(openvdb::Coord(coord)))
                        continue;
                    // this node is definitely outside the fluids
                    if(ssdf_axr.getValue(coord) > 0)
                        continue;
                    float pressValue = press_axr.getValue(openvdb::Coord(coord));
                    float Ax = 6 * pressValue;
                    for(int ss = 0; ss<3;++ss)
                    for(int i = -1;i <= 1;i += 2)
                    {
                        auto ipos = coord;
                        ipos[ss] += i;
                        if(!press_axr.isValueOn(openvdb::Coord(ipos)))
                        {
                            continue;
                        }
                        float press_value = press_axr.getValue(openvdb::Coord(ipos));
                        Ax -= press_value;
                    }
                    Ax /= dx * dx;
                    float b = rhs_axr.getValue(openvdb::Coord(coord));
                    res_axr.setValue(openvdb::Coord(coord), b - Ax);
                    p_axr.setValue(openvdb::Coord(coord), b - Ax);
                }
            }
        };

        auto computeRHS = [&](const tbb::blocked_range<size_t> &r) {
            auto rhs_axr{rhsGrid->getAccessor()};
            auto vel_axr{velGrid->getAccessor()};
            auto sdf_axr{sdfgrid->getConstAccessor()};
            auto ssdf_axr{staggeredSDFGrid->getAccessor()};
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelipos = leaf.offsetToGlobalCoord(offset);
                    if(pressGrid->tree().isValueOff(openvdb::Coord(voxelipos)))
                        continue;
                    float sdf = 0;
                    for(int i=-1;i<=0;++i)
                    for(int j=-1;j<=0;++j)
                    for(int k=-1;k<=0;++k)
                    {
                        auto ipos = voxelipos + openvdb::Coord(i,j,k);
                        sdf += sdf_axr.getValue(ipos);
                    }
                    sdf /= 8.0f;
                    ssdf_axr.setValue(voxelipos, sdf);
                    if(sdf>0)
                        continue;
                    
                    float divVel[3] = {0.0f, 0.0f, 0.0f};
                    int mask[3] = {1, 1, 1};
                    // x, y, z
                    float velV[6];
                    for(int ss = 0; ss<3;++ss)
                    for(int i = -1;i <= 0;i += 1)
                    {
                        float sum = 0;
                        int count = 0;
                        for(int j=-1;j<=0;++j)
                        for(int k=-1;k<=0;++k)
                        {
                            auto ipos = voxelipos;
                            ipos[ss] += i;
                            ipos[(ss + 1)%3] += j;
                            ipos[(ss + 2)%3] += k;
                            if(!vel_axr.isValueOn(openvdb::Coord(ipos)) || sdf_axr.getValue(openvdb::Coord(ipos)) > 0)
                            {
                                continue;
                            }
                            count++;
                            sum += vel_axr.getValue(openvdb::Coord(ipos))[ss];
                        }
                        // actually this will leads to numerical error!
                        // I believe using cut-cell method will be better
                        if(count != 0)
                            sum /= count;
                        else
                            mask[ss] = 0;
                        velV[ss * 2 + i + 1] = sum;
                    }
                    for(int ss = 0;ss<3;++ss)
                        divVel[ss] = mask[ss] * (velV[ss * 2 + 1] - velV[ss * 2]) / dx;
                    float rhsV = -(divVel[0] + divVel[1] + divVel[2])/dt;
                    rhs_axr.setValue(openvdb::Coord(voxelipos), rhsV);
                    if(rhsV != 0)
                        printf("rhs value is %f, sdf is %f\n", rhsV, sdf);
                }
            }
        };

        auto computeAlpha = [&](const tbb::blocked_range<size_t> &r, double alphaSum) {
            auto p_axr{pGrid->getAccessor()};
            auto res_axr{resGrid->getAccessor()};
            auto Ap_axr{ApGrid->getAccessor()};
            auto sdf_axr{sdfgrid->getConstAccessor()};
            auto ssdf_axr{staggeredSDFGrid->getConstAccessor()};
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelipos = leaf.offsetToGlobalCoord(offset);

                    if(pressGrid->tree().isValueOff(openvdb::Coord(voxelipos)))
                        continue;
                    if(ssdf_axr.getValue(voxelipos) > 0)
                        continue;
                    float pValue = p_axr.getValue(openvdb::Coord(voxelipos));
                    float Ap = 6 * pValue;
                    for(int ss = 0; ss<3;++ss)
                    for(int i = -1;i <= 1;i += 2)
                    {
                        auto ipos = voxelipos;
                        ipos[ss] += i;
                        if(pGrid->tree().isValueOff(openvdb::Coord(ipos)))
                            continue;
                        float press_value = p_axr.getValue(openvdb::Coord(ipos));
                        Ap -= press_value;
                    }
                    Ap /= dx * dx;
                    Ap_axr.setValue(openvdb::Coord(voxelipos), Ap);
                    float res = res_axr.getValue(openvdb::Coord(voxelipos));
                    //printf("res is %f\n", res);
                    if(pValue * Ap != 0)
                        alphaSum += res * res / (pValue * Ap);
                }
                return alphaSum;
            }
        };

        auto computeNewPress = [&](const tbb::blocked_range<size_t> &r) {
            auto res_axr{resGrid->getAccessor()};
            auto p_axr{pGrid->getAccessor()};
            auto press_axr = pressGrid->getAccessor();
            auto Ap_axr{ApGrid->getAccessor()};
            auto r2_axr{r2Grid->getAccessor()};
            auto sdf_axr{sdfgrid->getConstAccessor()};
            auto ssdf_axr{staggeredSDFGrid->getConstAccessor()};
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelipos = leaf.offsetToGlobalCoord(offset);

                    if(pressGrid->tree().isValueOff(openvdb::Coord(voxelipos)))
                        continue;
                    if(ssdf_axr.getValue(voxelipos) > 0)
                        continue;
                    float resValue = res_axr.getValue(openvdb::Coord(voxelipos));
                    float pValue = p_axr.getValue(openvdb::Coord(voxelipos));
                    float pressValue = press_axr.getValue(openvdb::Coord(voxelipos));
                    float ApValue = Ap_axr.getValue(openvdb::Coord(voxelipos));
                    press_axr.setValue(openvdb::Coord(voxelipos), pressValue + alpha * pValue);
                    r2_axr.setValue(openvdb::Coord(voxelipos), resValue - alpha * ApValue);
                }
            }
        };

        auto computeBeta1 = [&](const tbb::blocked_range<size_t> &r, float betaSum) {
            auto r2_axr{r2Grid->getAccessor()};
            auto res_axr{resGrid->getAccessor()};
            auto sdf_axr{sdfgrid->getConstAccessor()};
            auto ssdf_axr{staggeredSDFGrid->getConstAccessor()};
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelipos = leaf.offsetToGlobalCoord(offset);

                    if(pressGrid->tree().isValueOff(openvdb::Coord(voxelipos)))
                        continue;
                    if(ssdf_axr.getValue(voxelipos) > 0)
                        continue;
                    float r2Value = r2_axr.getValue(openvdb::Coord(voxelipos));
                    float rValue = res_axr.getValue(openvdb::Coord(voxelipos));
                    betaSum += r2Value * r2Value;
                    
                    //beta.fetch_add(r2Value * r2Value);
                    //alpha.fetch_add(rValue * rValue);
                }
            }
            return betaSum;
        };

        auto computeBeta2 = [&](const tbb::blocked_range<size_t> &r, float betaSum) {
            auto r2_axr{r2Grid->getAccessor()};
            auto res_axr{resGrid->getAccessor()};
            auto sdf_axr{sdfgrid->getConstAccessor()};
            auto ssdf_axr{staggeredSDFGrid->getConstAccessor()};
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelipos = leaf.offsetToGlobalCoord(offset);

                    if(pressGrid->tree().isValueOff(openvdb::Coord(voxelipos)))
                        continue;
                    if(ssdf_axr.getValue(voxelipos) > 0)
                        continue;
                    float r2Value = r2_axr.getValue(openvdb::Coord(voxelipos));
                    float rValue = res_axr.getValue(openvdb::Coord(voxelipos));
                    betaSum += rValue * rValue;
                    
                }
            }
            return betaSum;
        };

        auto computeP = [&](const tbb::blocked_range<size_t> &r) {
            auto r2_axr{r2Grid->getAccessor()};
            auto p_axr{pGrid->getAccessor()};
            auto res_axr{resGrid->getAccessor()};
            auto sdf_axr{sdfgrid->getConstAccessor()};
            auto ssdf_axr{staggeredSDFGrid->getConstAccessor()};
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelipos = leaf.offsetToGlobalCoord(offset);

                    if(pressGrid->tree().isValueOff(openvdb::Coord(voxelipos)))
                        continue;
                    if(ssdf_axr.getValue(voxelipos) > 0)
                        continue;
                    float r2Value = r2_axr.getValue(openvdb::Coord(voxelipos));
                    float pValue = p_axr.getValue(openvdb::Coord(voxelipos));
                    p_axr.setValue(openvdb::Coord(voxelipos), r2Value + beta * pValue);
                    res_axr.setValue(openvdb::Coord(voxelipos), r2Value);
                    
                }
            }
        };

        auto applyPress = [&](const tbb::blocked_range<size_t> &r){
            auto press_axr{pressGrid->getConstAccessor()};
            auto sdf_axr{sdfgrid->getAccessor()};
            auto vel_axr{velGrid->getAccessor()};
            //auto ssdf_axr{staggeredSDFGrid->getAccessor()};
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!sdf_axr.isValueOn(coord) || sdf_axr.getValue(coord) > 0)
                        continue;
                    // compute grad of press
                    openvdb::Vec3f gradPress(0.0f);
                    // x, y, z
                    float pressInter[6];
                    for(int ss = 0;ss<3;++ss)
                    for(int i=0;i<=1;++i)
                    {
                        pressInter[ss * 2 + i] = 0;
                        for(int j=0;j<=1;++j)
                        for(int k=0;k<=1;++k)
                        {
                            auto ipos = coord;
                            ipos[ss] += i;
                            ipos[(ss + 1)%3] += j;
                            ipos[(ss+2)%3] += k;
                            //the voxel is active and if outside, it's zero
                            float pV = press_axr.getValue(ipos);
                            pressInter[ss * 2 + i] += pV;
                        }
                        pressInter[ss * 2 + i] /= 4.0f;
                    }
                    for(int ss = 0;ss<3;++ss)
                        gradPress[ss] = (pressInter[ss * 2] - pressInter[ss * 2 + 1])/(dx*density)*dt;
                    vel_axr.setValue(coord, gradPress + vel_axr.getValue(coord));
                }
            }
        };

        // advection part 

        // velocity&&sdf extrapolation
        
        leaves.clear();
        tag->tree().getNodes(leaves);
        auto initTag = [&](const tbb::blocked_range<size_t> &r){
            auto tag_axr{tag->getAccessor()};
            auto sdf_axr{sdfgrid->getConstAccessor()};  
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto coord = leaf.offsetToGlobalCoord(offset);
                    if(!sdf_axr.isValueOn(coord))
                        continue;
                    auto sdfV = sdf_axr.getValue(coord);
                    if(sdfV>=0)
                        continue;
                    tag_axr.setValue(coord, 0);
                    for(int ss=0;ss<3;++ss)
                    for(int i=-1;i<=1;i+=2)
                    {
                        auto ipos = coord;
                        ipos[ss] += i;
                        if(!sdf_axr.isValueOn(ipos) || sdf_axr.getValue(ipos))
                        {
                            tag_axr.setValue(ipos, 1);
                        }
                    }
                }
            }
        
        };
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), initTag);
        
        auto newsdf = sdfgrid->deepCopy();
        auto newvel = velGrid->deepCopy();
        for(int d=1;d<5;++d)
        {
            auto computeTag = [&](const tbb::blocked_range<size_t> &r){
                auto tag_axr{tag->getAccessor()};
                auto sdf_axr{sdfgrid->getConstAccessor()};           
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        auto coord = leaf.offsetToGlobalCoord(offset);
                        if(!sdf_axr.isValueOn(coord))
                            continue;
                        int tag = tag_axr.getValue(coord);
                        if(tag == d - 1)
                        {                            
                            for(int ss=0;ss<3;++ss)
                            for(int i=-1;i<=1;i+=2)
                            {
                                auto ipos = coord;
                                ipos[ss] += i;
                                if(!tag_axr.isValueOn(ipos) || tag_axr.getValue(ipos) > d)
                                {
                                    tag_axr.setValue(ipos, d);
                                }
                            }
                        }
                                        
                    }
                }
            };
    
            auto sweep = [&](const tbb::blocked_range<size_t> &r){
                auto tag_axr{tag->getConstAccessor()};
                auto sdf_axr{sdfgrid->getConstAccessor()};
                auto vel_axr{velGrid->getAccessor()};
                
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        auto coord = leaf.offsetToGlobalCoord(offset);
                        if(!sdf_axr.isValueOn(coord))
                            continue;
                        int tagV = tag_axr.getValue(coord);
                        if(tagV != d)
                        {
                            continue;
                        }
                        openvdb::Vec3f vel(0.0f);
                        
                        int count = 0;
                        for(int ss = 0;ss < 3;++ss)
                        for(int i=-1;i<=1;i+=2)
                        {
                            auto ipos = coord;
                            ipos[ss] += i;
                            if(!tag_axr.isValueOn(ipos))
                                continue;
                            auto tagNei = tag_axr.getValue(ipos);
                            if(tagNei < tagV)
                            {
                                count++;
                                vel += vel_axr.getValue(ipos);
                            }
                        }
                        vel /= count;
                        vel_axr.setValue(coord, vel);
                    }
                }
            };

            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeTag);
            leaves.clear();
            tag->tree().getNodes(leaves);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), sweep);
        }
        
        // semi-lagrangian advection.
        {
            
            auto advect = [&](const tbb::blocked_range<size_t> &r){
                auto tag_axr{tag->getConstAccessor()};
                auto sdf_axr{sdfgrid->getConstAccessor()};
                auto vel_axr{velGrid->getConstAccessor()};
                auto nsdf_axr{newsdf->getAccessor()};
                auto nvel_axr{newvel->getAccessor()};
                //auto ssdf_axr{staggeredSDFGrid->getAccessor()};
                for (auto liter = r.begin(); liter != r.end(); ++liter) {
                    auto &leaf = *leaves[liter];
                    for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                        auto coord = leaf.offsetToGlobalCoord(offset);
                        auto wpos = sdfgrid->indexToWorld(coord);
                        if(!sdf_axr.isValueOn(coord) || tag_axr.getValue(coord) > 100)
                            continue;
                        auto vel = vel_axr.getValue(coord);
                        // at first we need to compute the mid-point velocity
                        auto midpos = openvdb::Vec3f(wpos - 0.5 * dt * vel);
                        auto tmp = sdfgrid->worldToIndex(midpos);
                        for(int ii=0;ii<3;++ii)
                            tmp[ii] = std::round(tmp[ii]);
                        openvdb::Coord midipos = openvdb::Coord(openvdb::Vec3i(tmp));
                        auto basepos = sdfgrid->indexToWorld(midipos);
                        openvdb::Vec3f midvel(0), x((midpos - basepos) / dx);
                        float weightsum = 0;
                        float neivels[8], nw[8];
                        for(int ii=0;ii<=1;++ii)
                        for(int jj=0;jj<=1;++jj)
                        for(int kk=0;kk<=1;++kk)
                        {
                            auto neipos = midipos + openvdb::Coord(ii, jj, kk);
                            if(!sdf_axr.isValueOn(neipos) || tag_axr.getValue(neipos) > 100)
                                continue;
                            auto neivel = vel_axr.getValue(neipos);
                            neivels[ii * 4 + jj * 2 + kk] = neivel[1];
                            
                            auto weight = (ii*(float)(x[0])+(1-ii)*(1-(float)(x[0])))*
                                            (jj*(float)(x[1])+(1-jj)*(1-(float)(x[1])))*
                                            (kk*(float)(x[2])+(1-kk)*(1-(float)(x[2])));
                            weightsum += weight;
                            nw[ii * 4 + jj * 2 + kk] = weight;
                            midvel += weight*neivel;
                        }
                        if(weightsum <= 0.01)
                            continue;
                        midvel /= weightsum;
                        weightsum = 0;
                        // then we can get the final position and its quantity
                        auto pwpos = openvdb::Vec3f(wpos - dt * midvel);
                        tmp = sdfgrid->worldToIndex(pwpos);
                        for(int ii=0;ii<3;++ii)
                            tmp[ii] = std::round(tmp[ii]);
                        openvdb::Coord pipos = openvdb::Coord(openvdb::Vec3i(tmp));
                        basepos = sdfgrid->indexToWorld(pipos);
                        openvdb::Vec3f pvel(0);
                        float psdf=0;
                        x = (pwpos - basepos) / dx;
                        
                        for(int ii=0;ii<=1;++ii)
                        for(int jj=0;jj<=1;++jj)
                        for(int kk=0;kk<=1;++kk)
                        {
                            auto neipos = midipos + openvdb::Coord(ii, jj, kk);
                            if(!sdf_axr.isValueOn(neipos) || tag_axr.getValue(neipos) > 100)
                                continue;
                            auto neivel = vel_axr.getValue(neipos);
                            auto neisdf = sdf_axr.getValue(neipos);
                            auto weight = (ii*(float)(x[0])+(1-ii)*(1-(float)(x[0])))*
                                            (jj*(float)(x[1])+(1-jj)*(1-(float)(x[1])))*
                                            (kk*(float)(x[2])+(1-kk)*(1-(float)(x[2])));
                            weightsum += weight;
                            pvel += weight * neivel;
                            psdf += weight * neisdf;
                        }
                          
                        pvel /= weightsum;
                        psdf /= weightsum;
                        nvel_axr.setValue(coord, pvel);
                        nsdf_axr.setValue(coord, psdf);
                        
                    }
                }
            };
            
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), advect);

        }
        data->sdf[0] = sdfgrid = newsdf->deepCopy();
        data->vel[0] = velGrid = newvel->deepCopy();
        data->cutOutGrid();

        //leaves.clear();
        //pressGrid->tree().getNodes(leaves);
        //printf("pressGrid leaves num is %d\n", leaves.size());
        
        // tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeRHS);

        // tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), initIter);
        
        // printf("start to solve pressure possision equation\n");
        // for(int iterNum = 0; iterNum < 2; ++iterNum)
        // {
        //     alpha = 0;
        //     beta = 0;
        //     alpha = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leaves.size()), 0.0,computeAlpha, std::plus<double>());
        //     tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeNewPress);
            
        //     beta = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leaves.size()), 0.0, computeBeta1, std::plus<float>());
        //     printf("iterNum is %d, beta is %f, alpha is %f, dx is %f\n", iterNum, beta, alpha, dx);
        //     if(beta < 0.0001 && beta > -0.0001)
        //         break;
        //     alpha = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leaves.size()), 0.0, computeBeta2, std::plus<float>());
        //     beta = beta / alpha;
        //     // assign r2 to r
        //     tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeP);
        // }
        // printf("pressure computation is done. beta is %f\n", beta);

        leaves.clear();
        sdfgrid->tree().getNodes(leaves);
        // //apply press
        // tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), applyPress);
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), applyGravityAndBound);
        tag->clear();
    };

};

ZENDEFNODE(AdaptiveSolver, {
        {"mgData", "dt", "density", "frameNum"},
        {"mgData"},
        {},
        {"AdaptiveSolver"},
});

struct selectLevelGrid : zeno::INode{
    virtual void apply() override {
        auto data = get_input<mgData>("mgData");
        int selectNum = has_input("level") ? get_input("level")->as<NumericObject>()->get<int>()
            :0;

        auto result = zeno::IObject::make<VDBFloatGrid>();
        result->m_grid = data->sdf[selectNum];
        //generate  
        set_output("vdbGrid", result);
    }
};
ZENDEFNODE(selectLevelGrid, {
        {"mgData", "level"},
        {"vdbGrid"},
        {},
        {"AdaptiveSolver"},
});

}