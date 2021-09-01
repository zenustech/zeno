#include "AdaptiveSolver.h"
#include <openvdb/Types.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
namespace zeno{

struct AdaptiveSolver : zeno::INode
{
    mgData data;
    float dt, density;
    virtual void apply() override {
        int levelNum = has_input("levelNum") ? 
            get_input("levelNum")->as<NumericObject>()->get<int>() : 1;
        auto level0 = get_input<VDBFloatGrid>("level0");
        auto level1 = has_input("level1") ? get_input<VDBFloatGrid>("level1")
            :zeno::IObject::make<VDBFloatGrid>();
        auto level2 = has_input("level2") ? get_input<VDBFloatGrid>("level2")
            :zeno::IObject::make<VDBFloatGrid>();
        auto level3 = has_input("level3") ? get_input<VDBFloatGrid>("level3")
            :zeno::IObject::make<VDBFloatGrid>();
        auto level4 = has_input("level4") ? get_input<VDBFloatGrid>("level4")
            :zeno::IObject::make<VDBFloatGrid>();
        float h = has_input("Dx") ? get_input("Dx")->as<NumericObject>()->get<float>()
            :0.08;
        float dt = has_input("dt") ? get_input("dt")->as<NumericObject>()->get<float>()
            :0.001;
        float density = has_input("density") ? get_input("density")->as<NumericObject>()->get<float>()
            :1000.0f;

        data.resize(levelNum);

        data.aig.topoLevels[0] = level0->m_grid;
        data.aig.topoLevels[1] = level1->m_grid;
        data.aig.topoLevels[2] = level2->m_grid;
        data.aig.topoLevels[3] = level3->m_grid;
        data.aig.topoLevels[4] = level4->m_grid;
        data.aig.hLevels[0] = h;
        
        data.initData();

        //generate  
        set_output("level0", level0);
        set_output("level1", level1);
        set_output("level2", level2);
        set_output("level3", level3);
        set_output("level4", level4);
    }

    //using cg iteration to solve press possion equation 
    void step()
    {
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        openvdb::FloatGrid::Ptr sdfgrid;
        openvdb::Vec3fGrid::Ptr velGrid = data.vel[data.aig.topoLevels.size()-1];
        openvdb::FloatGrid::Ptr pressGrid = data.press[data.aig.topoLevels.size()-1];
        // iteration terms
        openvdb::FloatGrid::Ptr rhsGrid = data.rhs[data.aig.topoLevels.size()-1];
        openvdb::FloatGrid::Ptr resGrid = data.residual[data.aig.topoLevels.size()-1];
        openvdb::FloatGrid::Ptr r2Grid = data.r2[data.aig.topoLevels.size()-1];
        
        openvdb::FloatGrid::Ptr pGrid = data.p[data.aig.topoLevels.size()-1];
        openvdb::FloatGrid::Ptr ApGrid = data.p[data.aig.topoLevels.size()-1];

        float dx = data.aig.hLevels[data.aig.topoLevels.size()-1];
        // compute the finest level only
        sdfgrid = data.aig.topoLevels[data.aig.topoLevels.size()-1];
        
        auto grid_axr{sdfgrid->getAccessor()};
        auto vel_axr{velGrid->getAccessor()};
        auto press_axr{pressGrid->getAccessor()};
        auto rhs_axr{rhsGrid->getAccessor()};
        auto res_axr{resGrid->getAccessor()};
        auto r2_axr{r2Grid->getAccessor()};
        auto p_axr{pGrid->getAccessor()};
        auto Ap_axr{ApGrid->getAccessor()};

        std::atomic<float> alpha;
        std::atomic<float> beta;
        
        // mark the points based on level set
        auto applyGravityAndBound = [&](const tbb::blocked_range<size_t> &r) {
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelwpos =
                        sdfgrid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                    auto voxelipos = openvdb::Vec3i(sdfgrid->worldToIndex(voxelwpos));

                    if(sdfgrid->tree().isValueOff(openvdb::Coord(voxelipos)))
                        continue;
                    openvdb::Vec3f vel_value = vel_axr.getValue(openvdb::Coord(voxelipos));
                    vel_value += openvdb::Vec3f(0, -dt * 9.8, 0);
                    // bound
                    if(voxelipos[1] <= -10)
                        vel_value = openvdb::Vec3f(vel_value[0], 0, vel_value[2]);
                    if(voxelipos[0] <= -30 || voxelipos[0] >= 30)
                        vel_value = openvdb::Vec3f(0, vel_value[1], vel_value[2]);
                    if(voxelipos[2] <= -30 || voxelipos[2] >= 30)
                        vel_value = openvdb::Vec3f(vel_value[0], vel_value[1], 0);

                    vel_axr.setValue(openvdb::Coord(voxelipos), vel_value);

                }
            }
        };

        auto computeRHS = [&](const tbb::blocked_range<size_t> &r) {
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelwpos =
                        pressGrid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                    auto voxelipos = openvdb::Vec3i(sdfgrid->worldToIndex(voxelwpos));

                    if(pressGrid->tree().isValueOff(openvdb::Coord(voxelipos)))
                        continue;
                    
                    float divVel = 0;
                    for(int ss = 0; ss<3;++ss)
                    for(int i = -1;i <= 1;i += 2)
                    {
                        auto ipos = voxelipos;
                        ipos[ss] += i;
                        if(velGrid->tree().isValueOff(openvdb::Coord(ipos)))
                            continue;
                        openvdb::Vec3f vel_value = vel_axr.getValue(openvdb::Coord(ipos));
                        divVel += i * vel_value[ss] / dx;
                    }
                    rhs_axr.setValue(openvdb::Coord(voxelipos), -divVel/dt);
                }
            }
        };

        // set r0 and p0
        auto initIter = [&](const tbb::blocked_range<size_t> &r) {
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelwpos =
                        pressGrid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                    auto voxelipos = openvdb::Vec3i(sdfgrid->worldToIndex(voxelwpos));

                    if(pressGrid->tree().isValueOff(openvdb::Coord(voxelipos)))
                        continue;
                    float pressValue = press_axr.getValue(openvdb::Coord(voxelipos));
                    float Ax = 6 * pressValue;
                    for(int ss = 0; ss<3;++ss)
                    for(int i = -1;i <= 1;i += 2)
                    {
                        auto ipos = voxelipos;
                        ipos[ss] += i;
                        if(pressGrid->tree().isValueOff(openvdb::Coord(ipos)))
                            continue;
                        float press_value = press_axr.getValue(openvdb::Coord(ipos));
                        Ax -= press_value;
                    }
                    Ax /= dx * dx;
                    float b = rhs_axr.getValue(openvdb::Coord(voxelipos));
                    res_axr.setValue(openvdb::Coord(voxelipos), b - Ax);
                    p_axr.setValue(openvdb::Coord(voxelipos), b - Ax);

                }
            }
        };

        auto computeAlpha = [&](const tbb::blocked_range<size_t> &r, float alphasum) {
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelwpos =
                        pressGrid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                    auto voxelipos = openvdb::Vec3i(sdfgrid->worldToIndex(voxelwpos));

                    if(pressGrid->tree().isValueOff(openvdb::Coord(voxelipos)))
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
                    if(pValue * Ap != 0)
                        alphaSum += res * res / (pValue * Ap);
                    return alphaSum;
                }
            }
        };

        auto computeNewPress = [&](const tbb::blocked_range<size_t> &r) {
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelwpos =
                        pressGrid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                    auto voxelipos = openvdb::Vec3i(sdfgrid->worldToIndex(voxelwpos));

                    if(pressGrid->tree().isValueOff(openvdb::Coord(voxelipos)))
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
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelwpos =
                        pressGrid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                    auto voxelipos = openvdb::Vec3i(sdfgrid->worldToIndex(voxelwpos));

                    if(pressGrid->tree().isValueOff(openvdb::Coord(voxelipos)))
                        continue;
                    float r2Value = r2_axr.getValue(openvdb::Coord(voxelipos));
                    float rValue = res_axr.getValue(openvdb::Coord(voxelipos));
                    betaSum += r2Value * r2Value;
                    return betaSum;
                    //beta.fetch_add(r2Value * r2Value);
                    //alpha.fetch_add(rValue * rValue);
                }
            }
        };

        auto computeBeta2 = [&](const tbb::blocked_range<size_t> &r, float betaSum) {
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelwpos =
                        pressGrid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                    auto voxelipos = openvdb::Vec3i(sdfgrid->worldToIndex(voxelwpos));

                    if(pressGrid->tree().isValueOff(openvdb::Coord(voxelipos)))
                        continue;
                    float r2Value = r2_axr.getValue(openvdb::Coord(voxelipos));
                    float rValue = res_axr.getValue(openvdb::Coord(voxelipos));
                    betaSum += rValue * rValue;
                    return betaSum;
                }
            }
        };

        auto computeP = [&](const tbb::blocked_range<size_t> &r) {
            // leaf iter
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    auto voxelwpos =
                        pressGrid->indexToWorld(leaf.offsetToGlobalCoord(offset));
                    auto voxelipos = openvdb::Vec3i(sdfgrid->worldToIndex(voxelwpos));

                    if(pressGrid->tree().isValueOff(openvdb::Coord(voxelipos)))
                        continue;
                    float r2Value = r2_axr.getValue(openvdb::Coord(voxelipos));
                    float pValue = p_axr.getValue(openvdb::Coord(voxelipos));
                    p_axr.setValue(openvdb::Coord(voxelipos), r2Value + beta * pValue);
                    res_axr.setValue(openvdb::Coord(voxelipos), r2Value);
                    
                }
            }
        };

        sdfgrid->tree().getNodes(leaves);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), applyGravityAndBound);
        
        leaves.clear();
        pressGrid->tree().getNodes(leaves);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeRHS);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), initIter);
        
        for(int iterNum = 0; iterNum < 10; ++iterNum)
        {
            alpha = 0;
            beta = 0;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeAlpha);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeNewPress);
            alpha = 0;
            beta = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leaves.size()), computeBeta1, std::plus<float>());
            if(beta < 0.0001 && beta > -0.0001)
                break;
            alpha = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leaves.size()), computeBeta2, std::plus<float>());
            beta = beta / alpha;
            // assign r2 to r
            tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeP);
        }
    };
};

ZENDEFNODE(AdaptiveSolver, {
        {"levelNum", "level0","level1","level2","level3", "level4", "Dx", "dt", "density"},
        {"level0","level1","level2","level3", "level4"},
        {},
        {"AdaptiveSolver"},
});


}