#include "UnifiedSolver.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace zeno {
#define BANKSIZE 32
#define DEFAULT_BLOCKSIZE 256
#define DEFAULT_WARPNUM 8

__device__ __forceinline__ unsigned int _LanemaskLt(int laneIdx) {
    return (1U << laneIdx) - 1;
}

template <typename T>
void UnifiedIPCSystem::SystemHessian<T>::initializePreconditioner(zs::CudaExecutionPolicy &pol,
                                                                  UnifiedIPCSystem &system) {
    int vertNum = system.numDofs;
    // compute nLevels, nTotalEntries
    int nLevel = 1;
    int levelSz = (vertNum + BANKSIZE - 1) / BANKSIZE * BANKSIZE;

    while (levelSz > BANKSIZE) {
        levelSz /= BANKSIZE;

        nLevel++;
        levelSz = (levelSz + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }

    levelnum = nLevel;
    totalNodes = vertNum;

    //fmt::print("checking vertNum: {}, {}, numLevel: {}\n", totalNodes, vertNum, levelnum);

    d_denseLevel = zs::Vector<int>{spmat.get_allocator(), (std::size_t)totalNodes};
    d_coarseTable = zs::Vector<zs::vec<int, 4>>{spmat.get_allocator(), (std::size_t)totalNodes};
    d_coarseSpaceTables = zs::Vector<int>{spmat.get_allocator(), (std::size_t)totalNodes * levelnum};
    d_levelSize = zs::Vector<zs::vec<int, 2>>{spmat.get_allocator(), (std::size_t)(levelnum + 1)};
    d_goingNext = zs::Vector<int>{spmat.get_allocator(), (std::size_t)(totalNodes * levelnum)};
    d_prefixOriginal = zs::Vector<int>{spmat.get_allocator(), (std::size_t)totalNodes};
    d_nextPrefix = zs::Vector<unsigned int>{spmat.get_allocator(), (std::size_t)totalNodes};
    d_nextPrefixSum = zs::Vector<unsigned int>{spmat.get_allocator(), (std::size_t)totalNodes};
    d_prefixSumOriginal = zs::Vector<int>{spmat.get_allocator(), (std::size_t)totalNodes};
    d_fineConnectMask = zs::Vector<unsigned int>{spmat.get_allocator(), (std::size_t)totalNodes};
    d_nextConnectMask = zs::Vector<unsigned int>{spmat.get_allocator(), (std::size_t)totalNodes};

    traversed = neighbors._ptrs;

    int nTotalEntries = ReorderRealtime(pol, 0);
    fmt::print("checking nTotalEntries: {}, {}\n", nTotalEntries, nTotalEntries);
    //
    Pm = zs::Vector<zs::vec<T, 96, 96>>{spmat.get_allocator(), (std::size_t)nTotalEntries / BANKSIZE};
    inversePm = zs::Vector<zs::vec<T, 96, 96>>{spmat.get_allocator(), (std::size_t)nTotalEntries / BANKSIZE};
    Rm = zs::Vector<zs::vec<T, 3>>{spmat.get_allocator(), (std::size_t)nTotalEntries};
    Zm = zs::Vector<zs::vec<T, 3>>{spmat.get_allocator(), (std::size_t)nTotalEntries};
}

template <typename T>
void UnifiedIPCSystem::SystemHessian<T>::BuildCollisionConnection(zs::CudaExecutionPolicy &pol,
                                                                  zs::Vector<unsigned int> &m_connectionMsk,
                                                                  zs::Vector<int> &m_coarseTableSpace, int level) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol(Collapse{dynHess.getCount()}, [dynHess = dynHess.port(), _pConnect = proxy<space>(m_connectionMsk),
                                       _pCoarseSpaceTable = proxy<space>(m_coarseTableSpace), level = level,
                                       totalNodes = totalNodes] __device__(int idx) mutable {
        auto nodeInex = zs::get<0>(dynHess[idx]);
        if (_pCoarseSpaceTable.data() != nullptr) {
            for (int i = 0; i < 2; i++)
                nodeInex[i] = _pCoarseSpaceTable[nodeInex[i] + (level - 1) * totalNodes];
        }
        unsigned int connMsk[2] = {0};

        unsigned int myId = nodeInex[0];
        unsigned int otId = nodeInex[1];

        if (myId / BANKSIZE == otId / BANKSIZE) {
            connMsk[0] = (1U << (otId % BANKSIZE));
            connMsk[1] = (1U << (myId % BANKSIZE));
        }

        for (int i = 0; i < 2; i++)
            atomicOr(&_pConnect[nodeInex[i]], connMsk[i]);
    });

#if 0
    pol(range(hess2.count()), [hess2 = proxy<space>(hess2), _pConnect = proxy<space>(m_connectionMsk),
                               _pCoarseSpaceTable = proxy<space>(m_coarseTableSpace), level = level,
                               totalNodes = totalNodes] __device__(int idx) mutable {
        auto nodeInex = hess2.inds[idx];
        if (_pCoarseSpaceTable.data() != nullptr) {
            for (int i = 0; i < 2; i++)
                nodeInex[i] = _pCoarseSpaceTable[nodeInex[i] + (level - 1) * totalNodes];
        }
        unsigned int connMsk[2] = {0};

        for (int i = 0; i < 2; i++) {
            for (int j = i + 1; j < 2; j++) {
                unsigned int myId = nodeInex[i];
                unsigned int otId = nodeInex[j];

                if (myId == otId) {
                    continue;
                }
                if (myId / BANKSIZE == otId / BANKSIZE) {
                    connMsk[i] |= (1U << (otId % BANKSIZE));
                    connMsk[j] |= (1U << (myId % BANKSIZE));
                }
            }
        }

        for (int i = 0; i < 2; i++)
            atomicOr(&_pConnect[nodeInex[i]], connMsk[i]);
    });

    pol(range(hess3.count()), [hess3 = proxy<space>(hess3), _connectionMsk = proxy<space>(m_connectionMsk),
                               _pCoarseSpaceTable = proxy<space>(m_coarseTableSpace), level = level,
                               totalNodes = totalNodes] __device__(int idx) mutable {
        auto nodeInex = hess3.inds[idx];
        if (_pCoarseSpaceTable.data() != nullptr) {
            for (int i = 0; i < 3; i++)
                nodeInex[i] = _pCoarseSpaceTable[nodeInex[i] + (level - 1) * totalNodes];
        }
        unsigned int connMsk[3] = {0};

        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 3; j++) {
                unsigned int myId = nodeInex[i];
                unsigned int otId = nodeInex[j];

                if (myId == otId) {
                    continue;
                }
                if (myId / BANKSIZE == otId / BANKSIZE) {
                    connMsk[i] |= (1U << (otId % BANKSIZE));
                    connMsk[j] |= (1U << (myId % BANKSIZE));
                }
            }
        }

        for (int i = 0; i < 3; i++)
            atomicOr(&_connectionMsk[nodeInex[i]], connMsk[i]);
    });

    pol(range(hess4.count()), [hess4 = proxy<space>(hess4), _connectionMsk = proxy<space>(m_connectionMsk),
                               _pCoarseSpaceTable = proxy<space>(m_coarseTableSpace), level = level,
                               totalNodes = totalNodes] __device__(int idx) mutable {
        auto nodeInex = hess4.inds[idx];
        if (_pCoarseSpaceTable.data() != nullptr) {
            for (int i = 0; i < 4; i++)
                nodeInex[i] = _pCoarseSpaceTable[nodeInex[i] + (level - 1) * totalNodes];
        }
        unsigned int connMsk[4] = {0};

        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                unsigned int myId = nodeInex[i];
                unsigned int otId = nodeInex[j];

                if (myId == otId) {
                    continue;
                }
                if (myId / BANKSIZE == otId / BANKSIZE) {
                    connMsk[i] |= (1U << (otId % BANKSIZE));
                    connMsk[j] |= (1U << (myId % BANKSIZE));
                }
            }
        }

        for (int i = 0; i < 4; i++)
            atomicOr(&_connectionMsk[nodeInex[i]], connMsk[i]);
    });
#endif
}

template <typename T>
int UnifiedIPCSystem::SystemHessian<T>::ReorderRealtime(zs::CudaExecutionPolicy &pol, int dynNum) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    //(cudaMemset(d_levelSize, 0, levelnum * sizeof(int2)));
    d_levelSize.reset(0);

    //BuildConnectMaskL0();

    fmt::print("num totalnodes: {}, fineconnectmask size: {}\n", totalNodes, d_fineConnectMask.size());
    pol(range(totalNodes), [_fineConnectedMsk = proxy<space>(d_fineConnectMask), neighbors = proxy<space>(neighbors),
                            traversed = proxy<space>(traversed)] __device__(int idx) mutable {
        int warpId = idx / 32;
        int laneId = idx % 32;
        //int numNeighbor = _neighborNum[idx];
        auto bg = neighbors._ptrs[idx];
        auto ed = neighbors._ptrs[idx + 1];
        int numNeighbor = ed - bg;

        unsigned int connectMsk = (1U << laneId);
        int nk = 0;
        int startId = traversed[idx]; //_neighborStart[idx];
        for (int i = 0; i < numNeighbor; i++) {
            int vIdConnected = neighbors._inds[startId + i]; //_neighborList[startId + i];
            if (vIdConnected == idx)
                continue;
            int warpIdxConnected = vIdConnected / BANKSIZE;
            if (warpId == warpIdxConnected) {
                unsigned int laneIdxConnected = vIdConnected % 32;
                connectMsk |= (1U << laneIdxConnected);
            } else {
                neighbors._inds[startId + nk] = vIdConnected;
                nk++;
            }
        }
        traversed[idx] = nk;
        _fineConnectedMsk[idx] = connectMsk;
    });

    if (dynNum) {
        zs::Vector<int> tmp{};
        BuildCollisionConnection(pol, d_fineConnectMask, tmp, -1);
    }
    //PreparePrefixSumL0();
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (totalNodes + blockSize - 1) / blockSize;

    pol(Collapse{numBlocks, blockSize},
        [vertNum = totalNodes, _fineConnectedMsk = proxy<space>(d_fineConnectMask),
         _prefixOriginal = proxy<space>(d_prefixOriginal)] __device__(int bno, int tid) mutable {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= vertNum)
                return;
            int warpId = idx / 32;
            int localWarpId = threadIdx.x / 32;
            int laneId = idx % 32;
            unsigned int connectMsk = _fineConnectedMsk[idx];
            //unsigned int connectMsk = cacheMask1;
            __shared__ int unsigned cacheMask[DEFAULT_BLOCKSIZE];
            __shared__ int prefixSum[DEFAULT_WARPNUM];
            if (laneId == 0) {
                prefixSum[localWarpId] = 0;
            }
            cacheMask[threadIdx.x] = connectMsk;
            unsigned int visited = (1U << laneId);
            while (connectMsk != -1) {
                unsigned int todo = visited ^ connectMsk;

                if (!todo)
                    break;

                unsigned int nextVist = __ffs(todo) - 1;
                visited |= (1U << nextVist);
                connectMsk |= cacheMask[nextVist + localWarpId * 32];
            }

            _fineConnectedMsk[idx] = connectMsk;

            unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));

            if (electedPrefix == 0) {
                //prefixSum[warpId]++;
                atomic_add(exec_cuda, prefixSum + localWarpId, 1);
            }

            if (laneId == 0) {
                _prefixOriginal[warpId] = prefixSum[localWarpId];
            }
        });

    //BuildLevel1();
    blockSize = BANKSIZE * BANKSIZE;
    numBlocks = (totalNodes + blockSize - 1) / blockSize;
    int warpNum = (totalNodes + 31) / 32;
    zs::exclusive_scan(pol, std::begin(d_prefixOriginal), std::begin(d_prefixOriginal) + warpNum,
                       std::begin(d_prefixSumOriginal));

    pol(Collapse{numBlocks, blockSize},
        [vertNum = totalNodes, _levelSize = proxy<space>(d_levelSize),
         _coarseSpaceTable = proxy<space>(d_coarseSpaceTables), _goingNext = proxy<space>(d_goingNext),
         _prefixOriginal = proxy<space>(d_prefixOriginal), _fineConnectedMsk = proxy<space>(d_fineConnectMask),
         _prefixSumOriginal = proxy<space>(d_prefixSumOriginal)] __device__(int bno, int tid) mutable {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= vertNum)
                return;
            auto mask = __activemask();
            mask = __ballot_sync(mask, 1);
            int warpId = idx / 32;
            int localWarpId = threadIdx.x / 32;
            int laneId = idx % 32;

            __shared__ unsigned int electedMask[BANKSIZE];
            if (laneId == 0) {
                electedMask[localWarpId] = 0;
            }
            if (idx == vertNum - 1) {
                _levelSize[1][0] = _prefixSumOriginal[warpId] + _prefixOriginal[warpId];
                _levelSize[1][1] = (vertNum + 31) / 32 * 32;
            }

            unsigned int connMsk = _fineConnectedMsk[idx];

            unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

            if (electedPrefix == 0) {
                atomicOr(electedMask + localWarpId, (1U << laneId));
            }

            unsigned int lanePrefix = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
            lanePrefix += _prefixSumOriginal[warpId];

            unsigned int elected_lane = __ffs(connMsk) - 1;
            unsigned int theLanePrefix = __shfl_sync(mask, lanePrefix, elected_lane);

            _coarseSpaceTable[idx + 0 * vertNum] = theLanePrefix;
            _goingNext[idx] = theLanePrefix + (vertNum + 31) / 32 * 32;
        });

    for (int level = 1; level < levelnum; level++) {
        d_nextConnectMask.reset(0);
        //BuildConnectMaskLx(level);
        blockSize = DEFAULT_BLOCKSIZE;
        numBlocks = (totalNodes + blockSize - 1) / blockSize;
        pol(Collapse{numBlocks, blockSize},
            [neighbors = proxy<space>(neighbors), traversed = proxy<space>(traversed),
             _fineConnectedMsk = proxy<space>(d_fineConnectMask), _coarseSpaceTable = proxy<space>(d_coarseSpaceTables),
             _nextConnectedMsk = proxy<space>(d_nextConnectMask), level = level,
             vertNum = totalNodes] __device__(int bno, int tid) mutable {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= vertNum)
                    return;
                int warpId = idx / 32;
                int localWarpId = threadIdx.x / 32;
                int laneId = idx % 32;

                unsigned int prefixMsk = _fineConnectedMsk[idx];
                unsigned int connMsk = 0;
                unsigned int coarseIdx = _coarseSpaceTable[(level - 1) * vertNum + idx];
                int kn = traversed[idx];
                int nk = 0;
                //auto bg = neighbors._ptrs[idx];
                int startId = neighbors._ptrs[idx];
                for (int i = 0; i < kn; i++) {
                    unsigned int connect = neighbors._inds[startId + i];
                    unsigned int coarseConnect = _coarseSpaceTable[(level - 1) * vertNum + connect];

                    if (coarseIdx / BANKSIZE == coarseConnect / BANKSIZE) {
                        unsigned int off = coarseConnect % BANKSIZE;
                        connMsk |= (1U << off);
                    } else {
                        neighbors._inds[startId + nk] = connect;
                        nk++;
                    }
                }

                traversed[idx] = nk;

                __shared__ int cacheMsk[DEFAULT_BLOCKSIZE];
                cacheMsk[threadIdx.x] = 0;

                if (prefixMsk == -1) {
                    atomicOr(cacheMsk + localWarpId * 32, connMsk);
                    connMsk = cacheMsk[localWarpId * 32];
                    //if (laneId == 0) {
                    //	cacheMsk[localWarpId] = 0;
                    //}
                } else {
                    unsigned int electedLane = __ffs(prefixMsk) - 1;
                    if (connMsk) {
                        atomicOr(cacheMsk + localWarpId * 32 + electedLane, connMsk);
                    }
                    connMsk = cacheMsk[localWarpId * 32 + electedLane];
                }

                unsigned int electedPrefix = __popc(prefixMsk & _LanemaskLt(laneId));

                if (connMsk && electedPrefix == 0) {
                    atomicOr(&_nextConnectedMsk[coarseIdx], connMsk);
                }
            });

        if (dynNum)
            BuildCollisionConnection(pol, d_nextConnectMask, d_coarseSpaceTables, level);

        // (cudaMemcpy(&h_clevelSize, d_levelSize.data() + level, sizeof(zs::vec<int, 2>), cudaMemcpyDeviceToHost));
        h_clevelSize = d_levelSize.getVal(level);

        //NextLevelCluster(level);
        numBlocks = (h_clevelSize[0] + blockSize - 1) / blockSize;
        pol(Collapse{numBlocks, blockSize},
            [vertNum = h_clevelSize[0], _nextConnectedMsk = proxy<space>(d_nextConnectMask),
             _nextPrefix = proxy<space>(d_nextPrefix), number = h_clevelSize[0]] __device__(int bno, int tid) mutable {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= number)
                    return;
                int warpId = idx / 32;
                int localWarpId = threadIdx.x / 32;
                int laneId = idx % 32;
                __shared__ int prefixSum[DEFAULT_WARPNUM];
                if (laneId == 0) {
                    prefixSum[localWarpId] = 0;
                }
                unsigned int connMsk = (1U << laneId);

                connMsk |= _nextConnectedMsk[idx];

                //unsigned int cachedMsk = connMsk;

                __shared__ unsigned int cachedMsk[DEFAULT_BLOCKSIZE];
                cachedMsk[threadIdx.x] = connMsk;
                unsigned int visited = (1U << laneId);

                while (true) {
                    unsigned int todo = visited ^ connMsk;

                    if (!todo)
                        break;

                    unsigned int nextVisit = __ffs(todo) - 1;

                    visited |= (1U << nextVisit);

                    connMsk |= cachedMsk[nextVisit + localWarpId * 32];
                }

                _nextConnectedMsk[idx] = connMsk;

                unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

                if (electedPrefix == 0) {
                    atomic_add(exec_cuda, prefixSum + localWarpId, 1);
                }

                if (laneId == 0)
                    _nextPrefix[warpId] = prefixSum[localWarpId];
            });

        //PrefixSumLx(level);
        blockSize = BANKSIZE * BANKSIZE;
        numBlocks = (h_clevelSize[0] + blockSize - 1) / blockSize;
        warpNum = (h_clevelSize[0] + 31) / 32;
        zs::exclusive_scan(pol, std::begin(d_nextPrefix), std::begin(d_nextPrefix) + warpNum,
                           std::begin(d_nextPrefixSum));
        pol(Collapse{numBlocks, blockSize},
            [_levelSize = proxy<space>(d_levelSize), _nextPrefix = proxy<space>(d_nextPrefix),
             _nextPrefixSum = proxy<space>(d_nextPrefixSum), _nextConnectMsk = proxy<space>(d_nextConnectMask),
             _goingNext = proxy<space>(d_goingNext), level = level, levelBegin = h_clevelSize[1],
             number = h_clevelSize[0]] __device__(int bro, int trid) mutable {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= number)
                    return;
                auto mask = __activemask();
                mask = __ballot_sync(mask, 1);
                int warpId = idx / 32;
                int localWarpId = threadIdx.x / 32;
                int laneId = idx % 32;

                __shared__ unsigned int electedMask[BANKSIZE];

                if (laneId == 0) {
                    electedMask[localWarpId] = 0;
                }

                if (idx == number - 1) {
                    _levelSize[level + 1][0] = _nextPrefixSum[warpId] + _nextPrefix[warpId];
                    _levelSize[level + 1][1] = levelBegin + (number + 31) / 32 * 32;
                }

                unsigned int connMsk = _nextConnectMsk[idx];

                unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

                if (electedPrefix == 0) {
                    atomicOr(electedMask + localWarpId, (1U << laneId));
                }

                unsigned int lanePrefix = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
                lanePrefix += _nextPrefixSum[warpId];

                unsigned int elected_lane = __ffs(connMsk) - 1;
                unsigned int theLanePrefix = __shfl_sync(mask, lanePrefix, elected_lane);

                _nextConnectMsk[idx] = theLanePrefix;
                _goingNext[idx + levelBegin] = theLanePrefix + levelBegin + (number + 31) / 32 * 32;
            });

        //ComputeNextLevel(level);
        pol(range(totalNodes),
            [_coarseSpaceTable = proxy<space>(d_coarseSpaceTables), _nextConnectMsk = proxy<space>(d_nextConnectMask),
             level = level, number = totalNodes] __device__(int idx) mutable {
                int next = _coarseSpaceTable[(level - 1) * number + idx];
                _coarseSpaceTable[(level)*number + idx] = _nextConnectMsk[next];
            });
    }

    h_clevelSize = d_levelSize.getVal(levelnum);

    totalNumberClusters = h_clevelSize[1];

    //AggregationKernel();
    pol(range(totalNodes),
        [levelNum = levelnum, _denseLevel = proxy<space>(d_denseLevel), _coarseTable = proxy<space>(d_coarseTable),
         _goingNext = proxy<space>(d_goingNext), levelnum = levelnum] __device__(int idx) mutable {
            auto mask = __activemask();
            mask = __ballot_sync(mask, 1);
            int currentId = idx;
            int aggLevel = levelNum - 1;
            zs::vec<int, 4> ctable;
            for (int l = 0; l < levelNum - 1; l++) {
                int next = _goingNext[currentId];
                int next0 = __shfl_sync(mask, next, 0);
                if (next == next0) {
                    aggLevel = (l < aggLevel ? l : aggLevel);
                }
                currentId = next;
                ctable[l] = next;
            }
            _denseLevel[idx] = aggLevel;
            _coarseTable[idx] = ctable;
        });
#if 0
    auto hctab = d_coarseTable.clone({memsrc_e::host, -1});
    for (auto v : hctab)
        fmt::print(fg(fmt::color::green), "{}, {}, {}, {}\n", v[0], v[1], v[2], v[3]);
#endif

    return totalNumberClusters;
}

template <typename T>
int UnifiedIPCSystem::SystemHessian<T>::buildPreconditioner(zs::CudaExecutionPolicy &pol, UnifiedIPCSystem &system) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    traversed = neighbors._ptrs;
    neighbors._inds = neighborInds;
    int dynNum = dynHess.getCount();
    ReorderRealtime(pol, dynNum);

    //(cudaMemset(Pm, 0, totalNumberClusters / BANKSIZE * sizeof(zs::vec<T, 96, 96>)));
    Pm.reset(0);

    //PrepareHessian(BH, masses);
    pol(range(totalNodes), [spmat = proxy<space>(spmat), _goingNext = proxy<space>(d_goingNext), Pm = proxy<space>(Pm),
                            levelnum = levelnum] __device__(int row) mutable {
        auto bg = spmat._ptrs[row];
        auto ed = spmat._ptrs[row + 1];
        auto orow = row;

        for (int i = bg; i < ed; i++) {
            row = orow;
            /// upper half
            int col = spmat._inds[i];
            int levelId = 0;
            while (row / 32 != col / 32 && levelId < levelnum) {
                levelId++;
                row = _goingNext[row];
                col = _goingNext[col];
            }

            if (levelId >= levelnum) {
                continue;
            }
            const auto mat = spmat._vals[i];
            int Pid = row / 32;
            auto &P = Pm[Pid];
            for (int j = 0; j < 3; j++) {
                for (int t = 0; t < 3; t++) {
                    atomic_add(exec_cuda, &P[(row % 32) * 3 + j][(col % 32) * 3 + t], mat[j][t]);
                }
            }
            while (levelId < levelnum - 1) {
                levelId++;
                row = _goingNext[row];
                col = _goingNext[col];
                Pid = row / 32;
                auto &P = Pm[Pid];
                for (int j = 0; j < 3; j++) {
                    for (int t = 0; t < 3; t++) {
                        atomic_add(exec_cuda, &P[(row % 32) * 3 + j][(col % 32) * 3 + t], mat[j][t]);
                    }
                }
            }
            if (i != bg) {
                /// lower half
                row = spmat._inds[i];
                col = orow;
                int levelId = 0;
                while (row / 32 != col / 32 && levelId < levelnum) {
                    levelId++;
                    row = _goingNext[row];
                    col = _goingNext[col];
                }

                if (levelId >= levelnum) {
                    continue;
                }
                int Pid = row / 32;
                auto &P = Pm[Pid];
                for (int j = 0; j < 3; j++) {
                    for (int t = 0; t < 3; t++) {
                        atomic_add(exec_cuda, &P[(row % 32) * 3 + j][(col % 32) * 3 + t], mat[t][j]);
                    }
                }

                while (levelId < levelnum - 1) {
                    levelId++;
                    row = _goingNext[row];
                    col = _goingNext[col];
                    Pid = row / 32;
                    auto &P = Pm[Pid];
                    for (int j = 0; j < 3; j++) {
                        for (int t = 0; t < 3; t++) {
                            atomic_add(exec_cuda, &P[(row % 32) * 3 + j][(col % 32) * 3 + t], mat[t][j]);
                        }
                    }
                }
            }
        }
    });

    int blockSize = DEFAULT_BLOCKSIZE;
    int number = dynNum;
    int numBlocks = (number + blockSize - 1) / blockSize;

    pol(Collapse{number}, [dynHess = dynHess.port(), _goingNext = proxy<space>(d_goingNext), Pm = proxy<space>(Pm),
                           number = number, levelNum = levelnum] __device__(int idx) mutable {
        auto [inds, mat] = dynHess[idx];
        int row = inds[0];
        int col = inds[1];

        int levelId = 0;
        while (row / 32 != col / 32 && levelId < levelNum) {
            levelId++;
            row = _goingNext[row];
            col = _goingNext[col];
        }
        if (levelId >= levelNum) {
            return;
        }
        int Pid = row / 32;

        auto &P = Pm[Pid];
        for (int t = 0; t < 3; t++) {
            for (int j = 0; j < 3; j++) {
                atomic_add(exec_cuda, &P[(row % 32) * 3 + t][(col % 32) * 3 + j], mat[t][j]);
                atomic_add(exec_cuda, &P[(col % 32) * 3 + t][(row % 32) * 3 + j], mat[j][t]);
            }
        }

        while (levelId < levelNum - 1) {
            levelId++;
            row = _goingNext[row];
            col = _goingNext[col];
            Pid = row / 32;
            auto &P = Pm[Pid];
            for (int t = 0; t < 3; t++) {
                for (int j = 0; j < 3; j++) {
                    atomic_add(exec_cuda, &P[(row % 32) * 3 + t][(col % 32) * 3 + j], mat[t][j]);
                    atomic_add(exec_cuda, &P[(col % 32) * 3 + t][(row % 32) * 3 + j], mat[j][t]);
                }
            }
        }
    });

    blockSize = 96 * 3;
    number = totalNumberClusters / BANKSIZE;
    number *= 96;
    numBlocks = (number + blockSize - 1) / blockSize;

#if 0
    fmt::print(fg(fmt::color::red), "before PmInverse compute, check Pm size {}, numTotalClusters: {}\n", number,
               totalNumberClusters);

    // check later version
    pol(Pm, [] __device__(auto &P) {
        for (int i = 0; i != 96; ++i)
            if (P(i, i) == 0)
                P(i, i) = 1;
    });
    PmBak = Pm.clone({memsrc_e::host, -1});
    inversePm = PmBak;

    auto ompPol = zs::omp_exec();
    ompPol(range(PmBak.size()), [&](int i) {
        using namespace Eigen;
        auto &Pm = PmBak[i]; // 96*96
        Matrix<double, 96, 96> mattest, invertest;
        for (int r = 0; r != 96; ++r)
            for (int c = 0; c != 96; ++c)
                mattest(r, c) = Pm(r, c);
        invertest = mattest.inverse();

        auto &invPm = inversePm[i];
        for (int r = 0; r != 96; ++r)
            for (int c = 0; c != 96; ++c)
                invPm(r, c) = invertest(r, c);
    });
    inversePmBak = inversePm;
    ompPol(range(PmBak.size()), [&](int i) {
        using namespace Eigen;
        auto &Pm = PmBak[i]; // 96*96
        Matrix<double, 96, 96> mattest, invertest;
        for (int r = 0; r != 96; ++r)
            for (int c = 0; c != 96; ++c)
                mattest(r, c) = Pm(r, c);
        invertest = mattest.inverse();

        auto &invPm = inversePm[i];
        for (int r = 0; r != 96; ++r)
            for (int c = 0; c != 96; ++c)
                invPm(r, c) = invertest(r, c);
    });
    std::atomic_int cnt = 0;
    std::vector<int> invalidPIndices(100);
    ompPol(enumerate(inversePm, inversePmBak), [&](int i, auto &P0, auto &P1) {
        if (std::isnan(P0(0, 0))) {
            auto id = cnt.fetch_add(1);
            if (id < invalidPIndices.size())
                invalidPIndices[id] = i;
        }
        for (int r = 0; r != 96; ++r)
            for (int c = 0; c != 96; ++c) {
                if (P0(r, c) != P1(r, c))
                    fmt::print("Pm inverse mismatch at P[{}] ({}, {}): {} <-> {}\n", i, r, c, P0(r, c), P1(r, c));
            }
    });
    if (cnt.load() != 0) {
        for (int k = 0; k != cnt.load(); ++k) {
            fmt::print("checking {}-th invalid Pm\n", invalidPIndices[k]);
            auto Pm = PmBak[invalidPIndices[k]];
            for (int r = 0; r != 96; ++r) {
                for (int c = 0; c != 96; ++c) {
                    fmt::print("{} ", Pm(r, c));
                }
                fmt::print("\n");
            }
        }
        exit(0);
    }

    inversePm = inversePm.clone({memsrc_e::device});

    auto inversePmRef = inversePm;
    PmBak = Pm;
#endif

    pol(Collapse{numBlocks, blockSize}, [P96 = proxy<space>(Pm), invP96 = proxy<space>(inversePm),
                                         number = number] __device__(int bro, int trid) mutable {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= number)
            return;
        int matId = idx / 96;
        int i = idx % 96;
        for (int j = 0; j < 96; j++) {
            if (i == j) {
                invP96[matId][j][i] = 1;
                if (P96[matId][j][i] == 0) {
                    P96[matId][j][i] = 1;
                }
            } else {
                invP96[matId][j][i] = 0;
            }
        }
        thread_fence(exec_cuda);
        __syncthreads();

        int j = 0;
        T rt = P96[matId][0][0];
        __syncthreads();
        while (j < 96) {
            if (i >= j) {
                P96[matId][j][i] /= rt;
            }
            invP96[matId][j][i] /= rt;
            thread_fence(exec_cuda);
            __syncthreads();

            for (int k = 0; k < 96; k++) {
                if (k != j) {
                    T rate = -P96[matId][k][j];
                    thread_fence(exec_cuda);
                    __syncthreads();
                    invP96[matId][k][i] += rate * invP96[matId][j][i];
                    if (i >= j) {
                        P96[matId][k][i] += rate * P96[matId][j][i];
                    }
                }
            }

            thread_fence(exec_cuda);
            __syncthreads();
            j++;
            rt = P96[matId][j][j];
        }
    });

#if 0
    pol(range(Pm.size()),
        [invP96 = proxy<space>(inversePm), invP96Ref = proxy<space>(inversePmRef)] __device__(int id) mutable {
            const auto &invP = invP96[id];       // conventional
            const auto &invPRef = invP96Ref[id]; // new
            for (int i = 0; i != 96; ++i)
                for (int j = 0; j != 96; ++j)
                    if (zs::abs(invP(i, j) - invPRef(i, j)) > limits<float>::epsilon() * 10) {
                        printf("divergence! invP[%d] at <%d, %d>: latter: %f, prev: %f, diff: %f\n", id, i, j,
                               (float)invP(i, j), (float)invPRef(i, j), (float)zs::abs(invP(i, j) - invPRef(i, j)));
                    }
        });
#endif

    return totalNumberClusters;
}

template <typename T>
void UnifiedIPCSystem::SystemHessian<T>::precondition(zs::CudaExecutionPolicy &pol, dtiles_t &vtemp,
                                                      const zs::SmallString srcTag, const zs::SmallString dstTag) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    pol.sync(true);
#if 1
    // Pm, Rm, Zm
    Rm.reset(0);
    Zm.reset(0);

    //pol(zs::range(numDofs), [cgtemp = proxy<space>({}, cgtemp), srcTag, dstTag] ZS_LAMBDA(int vi) mutable {
    //    cgtemp.template tuple<3>(dstTag, vi) =
    //        cgtemp.pack(dim_c<3, 3>, "P", vi) * cgtemp.pack(dim_c<3>, srcTag, vi);
    //});

    //BuildMultiLevelR(R);
    int blockSize = DEFAULT_BLOCKSIZE;
    int number = totalNodes;
    int numBlocks = (number + blockSize - 1) / blockSize;
    pol(Collapse{numBlocks, blockSize},
        [vtemp = proxy<space>({}, vtemp), srcTag, _multiLR = proxy<space>(Rm), _goingNext = proxy<space>(d_goingNext),
         _fineConnectMsk = proxy<space>(d_fineConnectMask), levelnum = levelnum, number] __device__(int, int) mutable {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= number)
                return;
            auto mask = __activemask();
            mask = __ballot_sync(mask, 1);
            auto _R = vtemp.pack(dim_c<3>, srcTag, idx);

            int laneId = threadIdx.x % 32;
            int localWarpId = threadIdx.x / 32;
            int level = 0;
            _multiLR[idx] = _R;

            __shared__ T c_sumResidual[DEFAULT_BLOCKSIZE * 3];

            unsigned int connectMsk = _fineConnectMsk[idx];
            if (connectMsk == -1) {
                for (int iter = 1; iter < 32; iter <<= 1) {
                    _R[0] += __shfl_down_sync(mask, _R[0], iter);
                    _R[1] += __shfl_down_sync(mask, _R[1], iter);
                    _R[2] += __shfl_down_sync(mask, _R[2], iter);
                }
                //int level = 0;

                if (laneId == 0) {
                    while (level < levelnum - 1) {
                        level++;
                        idx = _goingNext[idx];
                        atomic_add(exec_cuda, &_multiLR[idx][0], _R[0]);
                        atomic_add(exec_cuda, &_multiLR[idx][1], _R[1]);
                        atomic_add(exec_cuda, &_multiLR[idx][2], _R[2]);
                        //atomic_add(exec_cuda, (&((_multiLR + idx)->x)), r.x);
                        //atomic_add(exec_cuda, (&((_multiLR + idx)->x) + 1), r.y);
                        //atomic_add(exec_cuda, (&((_multiLR + idx)->x) + 2), r.z);
                    }
                }
                return;
            } else {
                int elected_lane = __ffs(connectMsk) - 1;

                c_sumResidual[threadIdx.x] = 0;
                c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE] = 0;
                c_sumResidual[threadIdx.x + 2 * DEFAULT_BLOCKSIZE] = 0;
                atomic_add(exec_cuda, c_sumResidual + localWarpId * 32 + elected_lane, _R[0]);
                atomic_add(exec_cuda, c_sumResidual + localWarpId * 32 + elected_lane + DEFAULT_BLOCKSIZE, _R[1]);
                atomic_add(exec_cuda, c_sumResidual + localWarpId * 32 + elected_lane + 2 * DEFAULT_BLOCKSIZE, _R[2]);

                unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));
                if (electedPrefix == 0) {
                    while (level < levelnum - 1) {
                        level++;
                        idx = _goingNext[idx];
                        atomic_add(exec_cuda, &_multiLR[idx][0], c_sumResidual[threadIdx.x]);
                        atomic_add(exec_cuda, &_multiLR[idx][1], c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE]);
                        atomic_add(exec_cuda, &_multiLR[idx][2], c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE * 2]);
                    }
                }
            }
        });

    //SchwarzLocalXSym();
    int matNum = totalNumberClusters / BANKSIZE;
    number = matNum * 96 * 32;
    blockSize = 96 * 3;
    numBlocks = (number + blockSize - 1) / blockSize;
    pol(Collapse{numBlocks, blockSize}, [P96 = proxy<space>(inversePm), dstTag, mR = proxy<space>(Rm),
                                         mZ = proxy<space>(Zm), number = number] __device__(int bro, int trid) mutable {
        namespace cg = ::cooperative_groups;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= number)
            return;

        auto tile = cg::tiled_partition<32>(cg::this_thread_block());

        int tileNo = idx / 32;
        int Hid = tileNo / 96;
        int MRid = tileNo % 96;

        int vrid = Hid * 32 + MRid / 3;
        auto laneid = tile.thread_rank();

        T sum = 0.;
        auto get_vcid = [Hid](int cid) { return Hid * 32 + cid / 3; };
        sum += P96[Hid][MRid][laneid] * mR[get_vcid(laneid)][laneid % 3];
        laneid += tile.num_threads();
        sum += P96[Hid][MRid][laneid] * mR[get_vcid(laneid)][laneid % 3];
        laneid += tile.num_threads();
        sum += P96[Hid][MRid][laneid] * mR[get_vcid(laneid)][laneid % 3];

        auto val = cg::reduce(tile, sum, cg::plus<T>());
        if (tile.thread_rank() == 0)
            mZ[vrid][MRid % 3] += val;
    });

    //CollectFinalZ(Z);
    pol(range(totalNodes),
        [vtemp = proxy<space>({}, vtemp), dstTag, d_multiLevelZ = proxy<space>(Zm),
         _coarseTable = proxy<space>(d_coarseTable), levelnum = levelnum] __device__(int idx) mutable {
            //zs::vec<T, 3> cz;
            auto cz = d_multiLevelZ[idx];

            auto table = _coarseTable[idx];
            //int* tablePtr = &(table.x);
            for (int i = 1; i < (levelnum < 4 ? levelnum : 4); i++) {
                int now = table[i - 1];
                cz += d_multiLevelZ[now];
            }

            vtemp.tuple(dim_c<3>, dstTag, idx) = cz;
        });

#endif
}

///
/// instantiations
///
template void
UnifiedIPCSystem::SystemHessian<typename UnifiedIPCSystem::T>::initializePreconditioner(zs::CudaExecutionPolicy &pol,
                                                                                        UnifiedIPCSystem &system);
template int
UnifiedIPCSystem::SystemHessian<typename UnifiedIPCSystem::T>::buildPreconditioner(zs::CudaExecutionPolicy &pol,
                                                                                   UnifiedIPCSystem &system);

template void UnifiedIPCSystem::SystemHessian<typename UnifiedIPCSystem::T>::precondition(
    zs::CudaExecutionPolicy &pol, typename UnifiedIPCSystem::dtiles_t &vtemp, const zs::SmallString srcTag,
    const zs::SmallString dstTag);

} // namespace zeno