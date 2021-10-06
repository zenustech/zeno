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
        //computeGradP(level, buffer.pGrid[level]);
        //computeDivP(level);
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
        //computeGradP(level, s);
        //computeDivP(level);
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
        //Propagate(level);
        Smooth(level);
    }
}
