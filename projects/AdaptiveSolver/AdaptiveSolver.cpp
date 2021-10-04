#include "AdaptiveSolver.h"
#include <openvdb/Types.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <openvdb/tools/LevelSetUtil.h>
namespace zeno{
void agData::computeRHS(int level)
{
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    pressField[level]->tree().getNodes(leaves);
    auto comRHS = [&](const tbb::blocked_range<size_t> &r)
    {
        auto status_axr = status[level]->getAccessor();
        auto rhs_axr = buffer.rhsGrid[level]->getAccessor();
        openvdb::tree::ValueAccessor<openvdb::FloatTree, true> vel_axr[3]=
            {velField[0][level]->getAccessor(),
            velField[1][level]->getAccessor(),
            velField[2][level]->getAccessor()};
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!status_axr.isValueOn(coord))
                    continue;
                auto s = status_axr.getValue(coord);
                if(s == 2)
                    continue;
                if(s == 1)
                {
                    rhs_axr.setValue(coord, 0);
                    continue;
                }
                float rhs = 0;
                for(int ss = 0;ss<3;++ss)
                for(int i=0;i<=1;i+=1)
                {
                    auto ipos = coord;
                    ipos[ss] += i;
                    float vel = vel_axr[ss].getValue(ipos);
                    rhs += (i-0.5)*2*vel/dx[level];
                }
                rhs_axr.setValue(coord, rhs);
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), comRHS);   

}

void agData::computeGradP(int level, openvdb::FloatGrid::Ptr p)
{
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    auto computeGradPress = [&](const tbb::blocked_range<size_t> &r)
    {
        auto status_axr = status[level]->getAccessor();
        auto press_axr = p->getAccessor();
        openvdb::tree::ValueAccessor<openvdb::FloatTree, true> gradP_axr[3]=
            {gradPressField[0][level]->getAccessor(),
            gradPressField[1][level]->getAccessor(),
            gradPressField[2][level]->getAccessor()};
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                for(int i=0;i<3;++i)
                {
                    if(!gradP_axr[i].isValueOn(coord))
                        continue;
                    float value = 0;
                    bool eqlevel = true;
                    for(int ss = -1;ss<=0;ss++)
                    {
                        auto ipos = coord;
                        ipos[i] += ss;
                        if(!status_axr.isValueOn(ipos) || status_axr.getValue(ipos) == 2)
                        {
                            value = 0;
                            break;
                        }
                        if(status_axr.getValue(ipos) == 1)
                        {
                            eqlevel = false;
                        }
                        value += (ss + 0.5) * 2 * press_axr.getValue(ipos);
                    }
                    if(eqlevel)
                        value /= dx[level];
                    else
                        value /= 1.5 * dx[level];
                    gradP_axr[i].setValue(coord, value);
                }
            }
        }
    };
    
    gradPressField[0][level]->tree().getNodes(leaves);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeGradPress);   
    leaves.clear();
}

void agData::computeDivP(int level)
{
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        pressField[level]->tree().getNodes(leaves);
    auto computeDivPress = [&](const tbb::blocked_range<size_t> &r)
    {
        auto status_axr = status[level]->getAccessor();
        auto ap_axr = buffer.ApGrid[level]->getAccessor();
        openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> gradP_axr[3]=
            {gradPressField[0][level]->getConstAccessor(),
            gradPressField[1][level]->getConstAccessor(),
            gradPressField[2][level]->getConstAccessor()};
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!status_axr.isValueOn(coord) || status_axr.getValue(coord) == 2)
                    continue;
                float divp = 0;
                for(int ss = 0;ss<3;++ss)
                for(int i=0;i<=1;i+=1)
                {
                    auto ipos = coord;
                    ipos[ss] += i;
                    
                    float gradp = gradP_axr[ss].getValue(ipos);
                    divp += (i-0.5)*2*gradp/dx[level];
                }
                ap_axr.setValue(coord, divp);
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeDivPress);   

}

void agData::comptueRES(int level, openvdb::FloatGrid::Ptr p)
{
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    
    auto computeRes = [&](const tbb::blocked_range<size_t> &r)
    {
        auto status_axr = status[level]->getAccessor();
        auto res_axr = buffer.resGrid[level]->getAccessor();
        auto rhs_axr = buffer.rhsGrid[level]->getConstAccessor();
        auto Ap_axr = buffer.ApGrid[level]->getAccessor();
        openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> gradP_axr[3]=
            {gradPressField[0][level]->getConstAccessor(),
            gradPressField[1][level]->getConstAccessor(),
            gradPressField[2][level]->getConstAccessor()};
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!status_axr.isValueOn(coord))
                    continue;
                auto s = status_axr.getValue(coord);
                if(s == 2)
                    continue;
                auto rhs = rhs_axr.getValue(coord);
                float lapP = Ap_axr.getValue(coord);
                res_axr.setValue(coord, rhs - lapP);
            }
        }
    };
    pressField[level]->tree().getNodes(leaves);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeRes);   
}

void agData::markGhost(){
    for(int level = 0;level<levelNum;++level)
    {
        std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
        pressField[level]->tree().getNodes(leaves);
        auto markG = [&](const tbb::blocked_range<size_t> &r)
        {
            auto status_axr = status[level]->getAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    if(!status_axr.isValueOn(coord))
                        continue;
                    if(status_axr.getValue(coord) != 2)
                        continue;
                    bool neiAC = false;
                    for(int ss = 0;ss<3;++ss)
                    for(int i=-1;i<=1;i+=2)
                    {
                        auto ipos = coord;
                        ipos[ss] += i;
                        if(!status_axr.isValueOn(ipos))
                            continue;
                        if(status_axr.getValue(coord) == 0)
                        {
                            neiAC = true;
                            break;
                        }
                    }
                    if(neiAC == false)
                        continue;
                    status_axr.setValue(coord, 1);
                }
            }
        };
    
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), markG);
        
    }
}

void agData::Smooth(int level)
{
    computeGradP(level, pressField[level]);
    computeDivP(level);
    // using damped jacobi
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    float d2 = 1 / (dx[level] * dx[level]);
    auto delta = pressField[level]->deepCopy();
    auto computeDelta = [&](const tbb::blocked_range<size_t> &r){
        auto status_axr = status[level]->getConstAccessor();
        auto press_axr = pressField[level]->getConstAccessor();
        auto rhs_axr = buffer.rhsGrid[level]->getConstAccessor();
        auto delta_axr = delta->getAccessor();
        auto ap_axr = buffer.ApGrid[level]->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!press_axr.isValueOn(coord))
                    continue;
                if(status_axr.getValue(coord) == 2)
                    continue;
                float ipress = press_axr.getValue(coord);
                float diag = 0, rhs = rhs_axr.getValue(coord);
                float lapP = ap_axr.getValue(coord);
                for(int ss = 0;ss<3;++ss)
                for(int i = -1;i<=1;i+=2)
                {
                    auto jcoord = coord;
                    jcoord[ss] += i;
                    if(!press_axr.isValueOn(jcoord) || status_axr.getValue(coord) == 2)
                        continue;
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
        auto status_axr = status[level]->getConstAccessor();
        auto press_axr = pressField[level]->getAccessor();
        auto delta_axr = delta->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!press_axr.isValueOn(coord))
                    continue;
                if(status_axr.getValue(coord) == 2)
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

void agData::Restrict(int level)
{
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    int coarse_level = level - 1;
    pressField[coarse_level]->tree().getNodes(leaves);
    auto Strict = [&](const tbb::blocked_range<size_t> &r)
    {
        auto status_axr = status[coarse_level]->getConstAccessor();
        auto fine_status_axr = status[level]->getConstAccessor();
        auto rhs_axr = buffer.rhsGrid[coarse_level]->getAccessor();
        auto res_axr = buffer.resGrid[level]->getConstAccessor();
        openvdb::tree::ValueAccessor<openvdb::FloatTree, true> vel_axr[3]=
            {velField[0][coarse_level]->getAccessor(),
            velField[1][coarse_level]->getAccessor(),
            velField[2][coarse_level]->getAccessor()};
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!status_axr.isValueOn(coord) || status_axr.getValue(coord) == 2)
                    continue;
                auto wpos = pressField[coarse_level]->indexToWorld(coord);
                auto baseIndex = openvdb::Vec3i(pressField[level]->worldToIndex(wpos));
                baseIndex[0] -= 1;baseIndex[1] -= 1;baseIndex[2] -= 1;
                float weight[3], sum = 0;
                for(int i=0;i<4;++i)
                {
                    weight[0] = i==0||i==3?1.0/8:3.0/8;
                    for(int j=0;j<4;++j)
                    {
                        weight[1] = j==0||j==3?1.0/8:3.0/8;
                        for(int k=0;k<4;++k)
                        {
                            auto ipos = openvdb::Coord(baseIndex) + openvdb::Coord(i, j, k);
                            if(!fine_status_axr.isValueOn(ipos) || fine_status_axr.getValue(ipos) == 2)
                                continue;
                            weight[2] = k==0||k==3?1.0/8:3.0/8;
                            float tw = weight[0] * weight[1] * weight[2];
                            sum += res_axr.getValue(ipos) * tw;
                        }
                    }
                }
                float rhs = 0;
                for(int ss = 0;ss<3;++ss)
                for(int i=0;i<=1;i+=1)
                {
                    auto ipos = coord;
                    ipos[ss] += i;
                    float vel = vel_axr[ss].getValue(ipos);
                    rhs += (i-0.5)*2*vel/dx[coarse_level];
                }

                rhs_axr.setValue(coord, sum + rhs);
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), Strict);
}

void agData::GhostValueAccumulate(int level)
{
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    int coarse_level = level - 1;
    pressField[coarse_level]->tree().getNodes(leaves);
    auto ghostValueAccumulate = [&](const tbb::blocked_range<size_t> &r)
    {
        auto status_axr = status[coarse_level]->getConstAccessor();
        auto fine_status_axr = status[level]->getConstAccessor();
        auto rhs_axr = buffer.rhsGrid[coarse_level]->getAccessor();
        auto res_axr = buffer.resGrid[level]->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!status_axr.isValueOn(coord) || status_axr.getValue(coord) == 2)
                    continue;
                auto wpos = pressField[coarse_level]->indexToWorld(coord);
                auto baseIndex = openvdb::Vec3i(pressField[level]->worldToIndex(wpos));
                int count = 0;
                float sum = 0;
                for(int i=0;i<=1;++i)
                for(int j=0;j<=1;++j)
                for(int k=0;k<=1;++k)
                {
                    auto ipos = openvdb::Coord(baseIndex) + openvdb::Coord(i,j,k);
                    if(fine_status_axr.getValue(ipos) == 1)
                    {
                        count++;
                        sum += res_axr.getValue(ipos);
                    }
                }
                if(count != 0)
                {
                    float rhs = rhs_axr.getValue(coord);
                    rhs_axr.setValue(coord, rhs + sum / count);
                }
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), ghostValueAccumulate);

}

void agData::GhostValuePropagate(int level)
{
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    int coarse_level = level - 1;
    pressField[coarse_level]->tree().getNodes(leaves);
    auto ghostValuePropagate = [&](const tbb::blocked_range<size_t> &r)
    {
        auto status_axr = status[coarse_level]->getConstAccessor();
        auto fine_status_axr = status[level]->getConstAccessor();
        auto fine_press_axr = pressField[level]->getAccessor();
        auto press_axr = pressField[coarse_level]->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!status_axr.isValueOn(coord) || status_axr.getValue(coord) == 2)
                    continue;
                auto wpos = pressField[coarse_level]->indexToWorld(coord);
                auto baseIndex = openvdb::Vec3i(pressField[level]->worldToIndex(wpos));
                float press = press_axr.getValue(coord);
                for(int i=0;i<=1;++i)
                for(int j=0;j<=1;++j)
                for(int k=0;k<=1;++k)
                {
                    auto ipos = openvdb::Coord(baseIndex) + openvdb::Coord(i,j,k);
                    if(fine_status_axr.getValue(ipos) == 1)
                    {
                        fine_press_axr.setValue(ipos, press);
                    }
                }
                
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), ghostValuePropagate);

}

void agData::Prolongate(int level)
{
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    int coarse_level = level - 1;
    pressField[level]->tree().getNodes(leaves);
    auto propagate = [&](const tbb::blocked_range<size_t> &r)
    {
        auto status_axr = status[coarse_level]->getConstAccessor();
        auto fine_status_axr = status[level]->getConstAccessor();
        auto fine_press_axr = pressField[level]->getAccessor();
        auto press_axr = pressField[coarse_level]->getConstAccessor();
        ConstBoxSample pressSampler(press_axr, pressField[coarse_level]->transform());
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!fine_status_axr.isValueOn(coord) || fine_status_axr.getValue(coord) > 0)
                    continue;
                auto wpos = pressField[coarse_level]->indexToWorld(coord);
                float press = pressSampler.wsSample(wpos);
                float fine_press = fine_press_axr.getValue(coord);
                fine_press_axr.setValue(coord, fine_press + press);
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), propagate);
}

float agData::dotTree(int level, openvdb::FloatGrid::Ptr a, openvdb::FloatGrid::Ptr b)
{
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    a->tree().getNodes(leaves);
    auto reduceAlpha = [&](const tbb::blocked_range<size_t> &r, float sum)
    {
        //auto press_axr
        auto a_axr = a->getConstAccessor();
        auto b_axr = b->getConstAccessor();
        auto status_axr = status[level]->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!status_axr.isValueOn(coord) || status_axr.getValue(coord) > 0)
                    continue;
                sum += a_axr.getValue(coord) * b_axr.getValue(coord);
            }
        }
        return sum;
    };
    float sum = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leaves.size()), 0.0f,reduceAlpha, std::plus<float>());
    return sum;
}
void agData::PossionSolver(int level)
{
    //BiCGSTAB solver
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    pressField[level]->tree().getNodes(leaves);
    computeGradP(level, pressField[level]);
    computeDivP(level);
    comptueRES(level, pressField[level]);
    auto r0 = buffer.resGrid[level]->deepCopy();

    float dens_pre = 1,alpha = 1, omg = 1;
    float dens, beta;
    buffer.pGrid[level]->clear();buffer.ApGrid[level]->clear();
    buffer.pGrid[level]->setTree((std::make_shared<openvdb::FloatTree>(
            pressField[level]->tree(), /*bgval*/ float(0),
            openvdb::TopologyCopy())));
    buffer.ApGrid[level]->setTree((std::make_shared<openvdb::FloatTree>(
            pressField[level]->tree(), /*bgval*/ float(0),
            openvdb::TopologyCopy())));
    
    auto s = buffer.pGrid[level]->deepCopy();
    auto h = buffer.pGrid[level]->deepCopy();
    
    auto computeP = [&](const tbb::blocked_range<size_t> &r)
    {
        //auto press_axr
        auto status_axr = status[level]->getConstAccessor();
        auto res_axr = buffer.resGrid[level]->getConstAccessor();
        auto p_axr = buffer.pGrid[level]->getAccessor();
        auto ap_axr = buffer.ApGrid[level]->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!status_axr.isValueOn(coord) || status_axr.getValue(coord) > 0)
                    continue;
                float p = p_axr.getValue(coord);
                float res = res_axr.getValue(coord);
                float ap = ap_axr.getValue(coord);
                p = res + beta *(p-omg * ap);
                p_axr.setValue(coord, p);
            }
        }
    };

    auto computeS = [&](const tbb::blocked_range<size_t> &r)
    {
        auto s_axr = s->getAccessor();
        auto status_axr = status[level]->getConstAccessor();
        auto res_axr = buffer.resGrid[level]->getConstAccessor();
        auto ap_axr = buffer.ApGrid[level]->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!status_axr.isValueOn(coord) || status_axr.getValue(coord) > 0)
                    continue;
                float s = res_axr.getValue(coord)- alpha * ap_axr.getValue(coord);
                s_axr.setValue(coord, s);
            }
        }
    };
    
    auto computeH = [&](const tbb::blocked_range<size_t> &r)
    {
        auto status_axr = status[level]->getConstAccessor();
        auto press_axr = pressField[level]->getConstAccessor();
        auto p_axr = buffer.pGrid[level]->getConstAccessor();
        auto h_axr = h->getAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!status_axr.isValueOn(coord) || status_axr.getValue(coord) > 0)
                    continue;
                float press = press_axr.getValue(coord);
                float p = p_axr.getValue(coord);
                h_axr.setValue(coord, press + alpha * p);
            }
        }
    };

    auto updatePress = [&](const tbb::blocked_range<size_t> &r)
    {
        auto press_axr = pressField[level]->getAccessor();
        auto h_axr = h->getConstAccessor();
        auto s_axr = s->getConstAccessor();
        auto status_axr = status[level]->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!status_axr.isValueOn(coord) || status_axr.getValue(coord) > 0)
                    continue;
                float hv = h_axr.getValue(coord);
                
                float sv = s_axr.getValue(coord);
                
                press_axr.setValue(coord, hv + omg * sv);
            }
        }
    };
 
    auto computeRes = [&](const tbb::blocked_range<size_t> &r)
    {
        auto res_axr = buffer.resGrid[level]->getAccessor();
        auto s_axr = s->getConstAccessor();
        auto status_axr = status[level]->getConstAccessor();
        auto ap_axr = buffer.ApGrid[level]->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!status_axr.isValueOn(coord) || status_axr.getValue(coord) > 0)
                    continue;
                float s = s_axr.getValue(coord);
                float ap = ap_axr.getValue(coord);
                float res = s - omg * ap;
                res_axr.setValue(coord, res);
            }
        }
    };

    auto reduceError = [&](const tbb::blocked_range<size_t> &r, float sum)
    {
        //auto press_axr
        auto ap_axr = buffer.ApGrid[level]->getConstAccessor();
        auto rhs_axr = buffer.rhsGrid[level]->getConstAccessor();
        auto status_axr = status[level]->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!status_axr.isValueOn(coord) || status_axr.getValue(coord) > 0)
                    continue;
                float ap = ap_axr.getValue(coord);
                float rhs = rhs_axr.getValue(coord);
                sum += abs(rhs - ap);
            }
        }
        return sum;
    };
    
    int iterCount = 0;
    do
    {
        dens = dotTree(level, r0, buffer.resGrid[level]);
        beta = dens * alpha / (dens_pre * omg);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeP);
        computeGradP(level, buffer.pGrid[level]);
        computeDivP(level);
        alpha = dotTree(level, r0, buffer.ApGrid[level]);
        alpha = dens / alpha;
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeH);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeS);
        float error = dotTree(level, s, s);
        if(error < 0.00001 * pressField[level]->activeVoxelCount())
        {
            pressField[level]->clear();
            pressField[level] = h->deepCopy();
            break;
        }
        computeGradP(level, s);
        computeDivP(level);
        omg = dotTree(level, s, buffer.ApGrid[level]) /  dotTree(level, buffer.ApGrid[level], buffer.ApGrid[level]);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), updatePress);
        
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeRes);
        error = dotTree(level, buffer.resGrid[level], buffer.resGrid[level]);
        if(error < 0.00001 * pressField[level]->activeVoxelCount())
            break;
        iterCount++;
        printf("iter num is %d, error is %f\n", iterCount, error);
    }
    while(iterCount < 20);
    //tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), propagate);

}

void agData::Vcycle(){
    computeRHS(levelNum-1);
    for(int level = levelNum-1;level>0;level--)
    {
        pressField[level]->clear();
        pressField[level]->setTree((std::make_shared<openvdb::FloatTree>(
            temperatureField[level]->tree(), /*bgval*/ float(0),
            openvdb::TopologyCopy())));
        Smooth(level);
        comptueRES(level, pressField[level]);
        Restrict(level);
        GhostValueAccumulate(level);
    }
    PossionSolver(0);
    for(int level = 1;level < levelNum;++level)
    {
        GhostValuePropagate(level);
        Propagate(level);
        Smooth(level);
    }
}
void agData::solvePress(){
    // as a preconditioner
    Vcycle();
    PossionSolver(levelNum - 1);
};
void agData::Advection()
{

}
void agData::step()
{
    applyOtherForce();
    solvePress();
    Advection();
}
struct AdaptiveSolver : zeno::INode
{
    std::shared_ptr<agData>  data;
    int frameNum;
    float dt, density, dx;
    float alpha;
    float beta;
    
    virtual void apply() override {
        data = get_input<agData>("agData");
        dt = has_input("dt") ? get_input("dt")->as<NumericObject>()->get<float>()
            :0.001;
        density = has_input("density") ? get_input("density")->as<NumericObject>()->get<float>()
            :1000.0f;
        frameNum = has_input("frameNum") ? get_input("frameNum")->as<NumericObject>()->get<int>()
            :0;
        
        data->step();
        //generate  
        set_output("agData", data);
    }
    

};

ZENDEFNODE(AdaptiveSolver, {
        {"agData", "dt", "density", "frameNum"},
        {"agData"},
        {},
        {"AdaptiveSolver"},
});


}