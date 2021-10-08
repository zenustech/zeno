#include "AdaptiveGridGen.h"
#include <tbb/parallel_reduce.h>

namespace zeno{
void agData::computeRHS()
{
    std::vector<openvdb::Int32Tree::LeafNodeType *> leaves;
    tag->tree().getNodes(leaves);
    auto comRHS = [&](const tbb::blocked_range<size_t> &r)
    {
        auto tag_axr = tag->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!tag_axr.isValueOn(coord))
                    continue;
                int level = tag_axr.getValue(coord);
                auto wpos = tag->indexToWorld(coord);

                auto lpos = round(status[level]->worldToIndex(wpos));
                auto status_axr = status[level]->getAccessor();
                if(!status_axr.isValueOn(lpos) || status_axr.getValue(lpos) != 0)
                    printf("error! tag doesn't match status\n");
                auto rhs_axr = buffer.rhsGrid[level]->getAccessor();
                openvdb::tree::ValueAccessor<openvdb::FloatTree, true> vel_axr[3]=
                    {velField[0][level]->getAccessor(),
                    velField[1][level]->getAccessor(),
                    velField[2][level]->getAccessor()};
                float rhs = 0;
                for(int ss = 0; ss< 3;++ss)
                for(int i=0;i<=1;++i)
                {
                    auto ipos = lpos;
                    ipos[ss] += i;
                    // dirichlet boundary
                    auto vel = vel_axr[ss].getValue(ipos);
                    rhs += (i-0.5)*2*vel/dx[level];
                }
                rhs_axr.setValue(lpos, rhs);
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), comRHS);   

}

void agData::computeLap(std::vector<openvdb::FloatGrid::Ptr> p, std::vector<openvdb::FloatGrid::Ptr> Ap)
{
    std::vector<openvdb::Int32Tree::LeafNodeType *> leaves;
    tag->tree().getNodes(leaves);

    auto comLap = [&](const tbb::blocked_range<size_t> &r)
    {
        auto tag_axr = tag->getAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!tag_axr.isValueOn(coord))
                    continue;
                int level = tag_axr.getValue(coord);
                auto wpos = tag->indexToWorld(coord);
                auto tmp = status[level]->worldToIndex(wpos);
                auto index = round(tmp);
                auto status_axr = status[level]->getAccessor();
                auto p_axr = p[level]->getConstAccessor();
                auto ap_axr = Ap[level]->getAccessor();
                float lapP = 0;
                // don't care about the symmetry things
                for(int ss= 0;ss<3;++ss)
                for(int i=-1;i<=1;i+=2)
                {
                    auto ipos = index;
                    ipos[ss] += i;
                    if(!status_axr.isValueOn(ipos))
                        continue;
                    int stat = status_axr.getValue(ipos);
                    if(stat == 2)
                    {
                        printf("computeLap error! stat is 2, level is %d, coord is (%d,%d,%d)\n",
                           level, ipos[0],ipos[1],ipos[2]);
                        break;
                    }
                    float p = p_axr.getValue(ipos);
                    lapP += p;
                }
                lapP -= 6 * p_axr.getValue(index);
                
                float pv = p_axr.getValue(index);
                auto pwpos = p[level]->indexToWorld(index);
                if(std::isnan(lapP) || std::isnan(pv) || std::isinf(pv) || std::isinf(lapP))
                    printf("p is %f, lapP is %f, level is %d, dx is %f, dt is %f, index is (%d,%d,%d),tmp is (%f,%f,%f) wpos is (%f,%f,%f), pwpos is (%f,%f,%f)\n", 
                        pv, lapP, level, dx[level],dt,
                        index[0],index[1],index[2],
                        tmp[0],tmp[1],tmp[2],
                        wpos[0],wpos[1],wpos[2],
                        pwpos[0],pwpos[1],pwpos[2]);
                lapP *= dt / (dx[level] * dx[level]);
                ap_axr.setValue(index, lapP);
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), comLap);   
}

void agData::comptueRES()
{
    std::vector<openvdb::Int32Tree::LeafNodeType *> leaves;
    tag->tree().getNodes(leaves);
    auto computeRes = [&](const tbb::blocked_range<size_t> &r)
    {
        auto tag_axr = tag->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!tag_axr.isValueOn(coord))
                    continue;
                int level = tag_axr.getValue(coord);
                auto wpos = tag->indexToWorld(coord);
                auto index = round(status[level]->worldToIndex(wpos));
                auto status_axr = status[level]->getConstAccessor();
                auto s = status_axr.getValue(index);
                if(s == 2)
                {    
                    printf("wrong tag on compute res\n");
                    continue;
                }
                auto res_axr = buffer.resGrid[level]->getAccessor();
                auto rhs_axr = buffer.rhsGrid[level]->getConstAccessor();
                auto Ap_axr = buffer.ApGrid[level]->getConstAccessor();
                auto rhs = rhs_axr.getValue(index);
                float lapP = Ap_axr.getValue(index);
                res_axr.setValue(index, rhs - lapP);
            }
        }
    };
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

float agData::dotTree(std::vector<openvdb::FloatGrid::Ptr> a, std::vector<openvdb::FloatGrid::Ptr> b)
{
    std::vector<openvdb::Int32Tree::LeafNodeType *> leaves;
    tag->tree().getNodes(leaves);
    auto reduceAlpha = [&](const tbb::blocked_range<size_t> &r, float sum)
    {
        //auto press_axr
        auto tag_axr = tag->getAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!tag_axr.isValueOn(coord))
                    continue;
                int level = tag_axr.getValue(coord);
                auto wpos = tag->indexToWorld(coord);
                auto lpos = round(status[level]->worldToIndex(wpos));
                auto a_axr = a[level]->getConstAccessor();
                auto b_axr = b[level]->getConstAccessor();
                sum += a_axr.getValue(lpos) * b_axr.getValue(lpos);
            }
        }
        return sum;
    };
    float sum = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, leaves.size()), 0.0f,reduceAlpha, std::plus<float>());
    return sum;
}

void agData::addTree(
            float alpha,float beta,
            std::vector<openvdb::FloatGrid::Ptr> a, 
            std::vector<openvdb::FloatGrid::Ptr> b,
            std::vector<openvdb::FloatGrid::Ptr> c)
{
    std::vector<openvdb::Int32Tree::LeafNodeType *> leaves;
    tag->tree().getNodes(leaves);
    auto add = [&](const tbb::blocked_range<size_t> &r)
    {
        auto tag_axr = tag->getAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!tag_axr.isValueOn(coord))
                    continue;
                
                int level = tag_axr.getValue(coord);
                auto wpos = tag->indexToWorld(coord);
                auto lpos = round(status[level]->worldToIndex(wpos));
                auto a_axr = a[level]->getConstAccessor();
                auto b_axr = b[level]->getConstAccessor();
                auto c_axr = c[level]->getAccessor();
                float av = a_axr.getValue(lpos);
                float bv = b_axr.getValue(lpos);
                c_axr.setValue(lpos, alpha * av + beta * bv);
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), add);
}

void agData::copyTree(std::vector<openvdb::FloatGrid::Ptr> a, std::vector<openvdb::FloatGrid::Ptr> b)
{
    for(int level = 0;level<levelNum;++level)
    {
        b[level]->clear();
        b[level] = a[level]->deepCopy();
    }
} 

void agData::PossionSolver()
{
    //BiCGSTAB solver
    std::vector<openvdb::Int32Tree::LeafNodeType *> leaves;
    tag->tree().getNodes(leaves);
    auto printPress = [&](const tbb::blocked_range<size_t> &r)
    {
        auto tag_axr = tag->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord tagcoord = leaf.offsetToGlobalCoord(offset);
                if(!tag_axr.isValueOn(tagcoord))
                    continue;
                int level = tag_axr.getValue(tagcoord);
                auto wpos = tag->indexToWorld(tagcoord);
                auto coord = round(status[level]->worldToIndex(wpos));

                auto press_axr = pressField[level]->getConstAccessor();
                auto press = press_axr.getValue(coord);
                //if(std::isnan(press))
                printf("coord (%d,%d,%d) press is %f\n",
                    coord[0], coord[1], coord[2], press);
            }
        }
    };
    
    computeLap(pressField, buffer.ApGrid);
    computeRHS();
    comptueRES();
    float dens_pre = 1,alpha = 1, omg = 1;
    float dens, beta, error;
    error = dotTree(buffer.resGrid, buffer.resGrid);
    if(error < 0.00001 * pressField[0]->activeVoxelCount())
    {
        printf("error is %f, exit iteration\n", error);
        return;
    }
    printf("error is %f, start to iter. active voxel count is %d\n", 
        error, pressField[0]->activeVoxelCount());
    
    std::vector<openvdb::FloatGrid::Ptr> r0, s, h, v, t;
    r0.resize(levelNum);s.resize(levelNum);h.resize(levelNum);
    v.resize(levelNum);t.resize(levelNum);
    for(int i=0;i<levelNum;++i)
    {
        r0[i] = buffer.resGrid[i]->deepCopy();
        buffer.ApGrid[i]->clear();
        buffer.ApGrid[i]->setTree((std::make_shared<openvdb::FloatTree>(
            pressField[i]->tree(), /*bgval*/ float(0),
            openvdb::TopologyCopy())));
        buffer.pGrid[i]->clear();
        buffer.pGrid[i]->setTree((std::make_shared<openvdb::FloatTree>(
            pressField[i]->tree(), /*bgval*/ float(0),
            openvdb::TopologyCopy())));
        s[i] = buffer.pGrid[i]->deepCopy();
        h[i] = buffer.pGrid[i]->deepCopy();
        v[i] = buffer.pGrid[i]->deepCopy();
        t[i] = buffer.pGrid[i]->deepCopy();
    }
    
    auto computeP = [&](const tbb::blocked_range<size_t> &r)
    {
        //auto press_axr
        auto tag_axr = tag->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord tagcoord = leaf.offsetToGlobalCoord(offset);
                if(!tag_axr.isValueOn(tagcoord))
                    continue;
                int level = tag_axr.getValue(tagcoord);
                auto wpos = tag->indexToWorld(tagcoord);
                auto coord = round(status[level]->worldToIndex(wpos));

                auto res_axr = buffer.resGrid[level]->getConstAccessor();
                auto p_axr = buffer.pGrid[level]->getAccessor();
                auto ap_axr = v[level]->getConstAccessor();

                float p = p_axr.getValue(coord);
                float res = res_axr.getValue(coord);
                float ap = ap_axr.getValue(coord);
                p = res + beta * (p - omg * ap);
                p_axr.setValue(coord, p);
            }
        }
    };

    int iterCount = 0;
    do
    {
        dens = dotTree(r0, buffer.resGrid);
        beta = dens * alpha / (dens_pre * omg);
        //printf("dens is %f, beta is %f\n", dens, beta);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), computeP);
        
        transferPress(buffer.pGrid);
        // store v on apgrid
        computeLap(buffer.pGrid, v);
        alpha = dotTree(r0, v);
        alpha = dens / alpha;

        addTree(1, alpha, pressField, buffer.pGrid, h);
        addTree(1, -alpha, buffer.resGrid, v, s);
        
        error = dotTree(s, s);
        //printf("alpha is %f, error is %f\n", alpha, error);
        
        if(error < 0.00001 * pressField[0]->activeVoxelCount())
        {
            copyTree(h, pressField);
            break;
        }
        transferPress(s);
        
        computeLap(s, t);
        
        omg = dotTree(s, t) /  dotTree(t, t);
        addTree(1, omg, h, s, pressField);
        
        addTree(1, -omg, s, t, buffer.resGrid);
        error = dotTree(buffer.resGrid, buffer.resGrid);
        //printf("omg is %f, error is %f\n", omg, error);
        if(error < 0.00001 * pressField[0]->activeVoxelCount())
            break;
        iterCount++;
        //printf("iter num is %d, error is %f\n", iterCount, error);
        dens_pre = dens;
    }
    while(iterCount < 20);
    printf("iter over. error is %f\n", error);
    //tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), propagate);
    //tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), printPress);
     
    for(int i=0;i<levelNum;++i)
    {
        r0[i]->clear();
        s[i]->clear();
        h[i]->clear();
    }
}

void agData::transferPress(std::vector<openvdb::FloatGrid::Ptr> p)
{
    for(int level = 1; level < levelNum;++level)
    {
        std::vector<openvdb::Int32Tree::LeafNodeType *> leaves;
        status[level]->tree().getNodes(leaves);
        auto TopDown = [&](const tbb::blocked_range<size_t> &r)
        {
            auto status_axr = status[level]->getConstAccessor();
            auto press_axr = p[level]->getAccessor();
            auto fine_press_axr = p[level-1]->getConstAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    if(!status_axr.isValueOn(coord) || status_axr.getValue(coord) == 0)
                        continue;
                    auto wpos = status[level]->indexToWorld(coord);
                    float psum = 0;
                    for(int i=-1;i<=1;i+=2)
                    for(int j=-1;j<=1;j+=2)
                    for(int k=-1;k<=1;k+=2)
                    {
                        auto nwpos = wpos + openvdb::Vec3d(i,j,k) * 0.25 * dx[level];
                        auto ncoord = round(status[level-1]->worldToIndex(nwpos));
                        psum += fine_press_axr.getValue(ncoord);
                    }
                    press_axr.setValue(coord, psum / 8);
                }
            }

        };
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), TopDown);
    }
    for(int level = levelNum-2;level>=0;--level)
    {
        std::vector<openvdb::Int32Tree::LeafNodeType *> leaves;
        status[level]->tree().getNodes(leaves);
        auto DownTop = [&](const tbb::blocked_range<size_t> &r)
        {
            auto status_axr = status[level]->getConstAccessor();
            auto press_axr = p[level]->getAccessor();
            auto coarse_press_axr = p[level+1]->getConstAccessor();
            for (auto liter = r.begin(); liter != r.end(); ++liter) {
                auto &leaf = *leaves[liter];
                for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                    openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                    if(!status_axr.isValueOn(coord) || status_axr.getValue(coord) == 0)
                        continue;
                    auto wpos = status[level]->indexToWorld(coord);
                    auto cindex = round(status[level+1]->worldToIndex(wpos));
                    auto press = coarse_press_axr.getValue(cindex);
                    press_axr.setValue(coord, press);
                }
            }
        };
        tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), DownTop);
    }
}

void agData::applyPress()
{
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    velField[0][0]->tree().getNodes(leaves);
    auto updateVel = [&](const tbb::blocked_range<size_t> &r)
    {
        openvdb::tree::ValueAccessor<openvdb::FloatTree, true> vel_axr[3]=
            {velField[0][0]->getAccessor(),
            velField[1][0]->getAccessor(),
            velField[2][0]->getAccessor()};
        auto press_axr = pressField[0]->getConstAccessor();
        auto status_axr = status[0]->getConstAccessor();
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                for(int i=0;i<3;++i)
                {
                    if(!vel_axr[i].isValueOn(coord))
                        continue;
                    bool canApply = true;
                    float vel = vel_axr[i].getValue(coord);
                    float gradP = 0;
                    for(int j=0;j<=1;++j){
                        auto ipos = coord;
                        ipos[i] += j;
                        gradP += (j-0.5)*2*press_axr.getValue(ipos);
                    }
                    
                    vel -= dt * gradP /(dx[0] * dens);
                    float mol = abs(vel);
                    float maxvel = 1;
                    if(mol > maxvel)
                        vel = vel / mol * maxvel;
                    vel_axr[i].setValue(coord, vel);
                }
            }
        }   
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), updateVel);
}
void agData::solvePress(){
    //todo: add multigrid as a preconditioner
    makeCoarse();
    PossionSolver();
    transferPress(pressField);

    applyPress();
};
void agData::Advection()
{
    // only advect the finest level grid
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    volumeField[0]->tree().getNodes(leaves);
    int level = 0;
    int sign = 1;
    auto new_temField = temperatureField[0]->deepCopy();
    auto new_volField = volumeField[0]->deepCopy();
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
                if(volume < 0.001)
                    volume = 0;
                if(tem < 0.001)
                    tem = 0;
                
                new_tem_axr.setValue(coord, tem);
                new_vol_axr.setValue(coord, volume);
            }
        }
    };

    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), semiLangAdvection);
    temperatureField[0]->clear();
    volumeField[0]->clear();
    temperatureField[0] = new_temField;
    volumeField[0] = new_volField;

    leaves.clear();
    velField[0][0]->tree().getNodes(leaves);
    openvdb::FloatGrid::Ptr new_vel[3], inte_vel[3];
    for(int i=0;i<3;++i)
    {
        new_vel[i] = velField[i][0]->deepCopy();
        inte_vel[i] = velField[i][0];
    }
    auto velAdvection = [&](const tbb::blocked_range<size_t> &r){
        openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> 
            vel_axr[3]=
            {velField[0][0]->getConstAccessor(),
            velField[1][0]->getConstAccessor(),
            velField[2][0]->getConstAccessor()};
        openvdb::tree::ValueAccessor<const openvdb::FloatTree, true> 
            inte_axr[3]=
            {inte_vel[0]->getConstAccessor(),inte_vel[1]->getConstAccessor(),inte_vel[2]->getConstAccessor()};
        openvdb::tree::ValueAccessor<openvdb::FloatTree, true> 
            new_vel_axr[3]=
            {new_vel[0]->getAccessor(),new_vel[1]->getAccessor(),new_vel[2]->getAccessor()};
        ConstBoxSample velSampler1[3] = {
            ConstBoxSample(vel_axr[0], velField[0][level]->transform()),
            ConstBoxSample(vel_axr[1], velField[1][level]->transform()),
            ConstBoxSample(vel_axr[2], velField[2][level]->transform())
        };
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
                        vel[j] = velSampler1[j].wsSample(wpos);
                
                    auto midwpos = wpos - sign *0.5 * dt * vel;
                    for(int j=0;j<3;++j)
                        midvel[j] = velSampler1[j].wsSample(midwpos);
                    auto pwpos = wpos - sign * dt * midvel;
                    auto pvel = velSampler[i].wsSample(pwpos);
                    new_vel_axr[i].setValue(coord, pvel);
                }
            }
        }

    };
    
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), velAdvection);
    for(int i=0;i<3;++i)
    {
        velField[i][0]->clear();
        velField[i][0] = new_vel[i];
    }
}

void agData::applyOtherForce()
{
    float alpha =-0.1, beta = 0.2;
    std::vector<openvdb::FloatTree::LeafNodeType *> leaves;
    velField[1][0]->tree().getNodes(leaves);
    auto applyOtherForce = [&](const tbb::blocked_range<size_t> &r)
    {
        openvdb::tree::ValueAccessor<openvdb::FloatTree, true> vel_axr[3]=
            {velField[0][0]->getAccessor(),
            velField[1][0]->getAccessor(),
            velField[2][0]->getAccessor()};
        auto status_axr = status[0]->getConstAccessor();
        auto tem_axr = temperatureField[0]->getConstAccessor();
        auto vol_axr = volumeField[0]->getConstAccessor();
        ConstBoxSample volSample(vol_axr, volumeField[0]->transform());
        ConstBoxSample temSample(tem_axr, temperatureField[0]->transform());
                
        for (auto liter = r.begin(); liter != r.end(); ++liter) {
            auto &leaf = *leaves[liter];
            for (auto offset = 0; offset < leaf.SIZE; ++offset) {
                openvdb::Coord coord = leaf.offsetToGlobalCoord(offset);
                if(!vel_axr[1].isValueOn(coord))
                    continue;
                auto wpos = velField[1][0]->indexToWorld(coord);
                auto vol = volSample.wsSample(wpos);
                auto tem = temSample.wsSample(wpos);

                float voldens = (alpha * vol -beta * tem);
                auto vel = vel_axr[1].getValue(coord);
                auto deltaV = vel - voldens * 9.8 * dt;
                vel_axr[1].setValue(coord, deltaV);
                //if(deltaV != 0)
                //    printf("apply force, deltaV is %f\n", deltaV);
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), applyOtherForce);
    
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