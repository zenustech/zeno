#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/LevelSetUtils.tpp"
#include "zensim/geometry/SparseGrid.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

#include <zeno/VDBGrid.h>

#include "zensim/Logger.hpp"
#include <zeno/utils/log.h>

#include "../scheme.hpp"
#include "../utils.cuh"

namespace zeno {

#define ENABLE_PROFILE 0

struct ZSNSPressureProject : INode {

#if ENABLE_PROFILE
    inline static float s_smoothTime[4];
    inline static float s_restrictTime[4];
    inline static float s_prolongateTime[4];
    inline static float s_coarseTime;
    inline static float s_residualTime[4];
    inline static float s_coarseResTime;
    inline static zs::CppTimer s_timer, s_timer_p, s_timer_m;
    inline static float s_rhsTime, s_projectTime, s_mginitTime, s_multigridTime, s_totalTime;
#endif

#if 0
    template <int level, typename Ti>
    void PostJacobi(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid, Ti block_cnt, int cur) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;
        auto &spg = NSGrid->getLevel<level>();
        // workaround: write result to "p0" for high level grid
        if (cur == 0)
            return;
        pol(zs::range(block_cnt * spg.block_size),
            [spgv = zs::proxy<space>(spg),
             pSrcTag = zs::SmallString{std::string("p") + std::to_string(cur)}] __device__(int cellno) mutable {
                auto icoord = spgv.iCoord(cellno);
                spgv("p0", icoord) = spgv(pSrcTag, icoord);
            });
    }

    template <int level> void Jacobi(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid, float rho, int nIter) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->getLevel<level>();
        auto block_cnt = spg.numBlocks();

        auto dx = spg.voxelSize()[0];

        int cur = level == 0 ? NSGrid->readMeta<int>("p_cur") : 0;

        for (int iter = 0; iter < nIter; ++iter) {
            // point Jacobi iteration
            pol(zs::range(block_cnt * spg.block_size),
                [spgv = zs::proxy<space>(spg), dx, rho,
                 pSrcTag = zs::SmallString{std::string("p") + std::to_string(cur)},
                 pDstTag = zs::SmallString{std::string("p") + std::to_string(cur ^ 1)}] __device__(int cellno) mutable {
                    auto icoord = spgv.iCoord(cellno);

                    float div = spgv.value("tmp", icoord);

                    const int stcl = 1; // stencil point in each side
                    float p_x[2 * stcl + 1], p_y[2 * stcl + 1], p_z[2 * stcl + 1];

                    for (int i = -stcl; i <= stcl; ++i) {
                        p_x[i + stcl] = spgv.value(pSrcTag, icoord + zs::vec<int, 3>(i, 0, 0));
                        p_y[i + stcl] = spgv.value(pSrcTag, icoord + zs::vec<int, 3>(0, i, 0));
                        p_z[i + stcl] = spgv.value(pSrcTag, icoord + zs::vec<int, 3>(0, 0, i));
                    }

                    float p_this = -(div * dx * dx * rho - (p_x[0] + p_x[2] + p_y[0] + p_y[2] + p_z[0] + p_z[2])) / 6.f;

                    spgv(pDstTag, icoord) = p_this;
                });
            cur ^= 1;
        }
        if constexpr (level == 0)
            NSGrid->setMeta("p_cur", cur);
        else {
            PostJacobi<level>(pol, NSGrid, block_cnt, cur);
        }
    }
#endif

    template <int level>
    void clearInit(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->getLevel<level>();
        auto block_cnt = spg.numBlocks();

        // take zero as initial guess
        pol(zs::range(block_cnt * spg.block_size),
            [spgv = zs::proxy<space>(spg), pOffset = spg.getPropertyOffset("p")] __device__(int cellno) mutable {
                spgv(pOffset, cellno) = 0.f;
            });
    }

    template <int level>
    void redblackSOR(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid, float rho, float sor, int nIter) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->getLevel<level>();
        auto block_cnt = spg.numBlocks();

        auto dx = spg.voxelSize()[0];

        pol.sync(false);
        for (int iter = 0; iter < nIter; ++iter) {
            // a simple implementation of red-black SOR
            for (int clr = 0; clr != 2; ++clr) {
#if 1
                using value_type = typename RM_CVREF_T(spg)::value_type;
                constexpr int side_length = RM_CVREF_T(spg)::side_length;
                static_assert(side_length == 1, "coarsest level block assumed to have a side length of 1.");
                // 7 * sizeof(float) = 28 Bytes / block
                constexpr std::size_t bucket_size = RM_CVREF_T(spg._table)::bucket_size;
                constexpr std::size_t tpb = 6;
                constexpr std::size_t cuda_block_size = bucket_size * tpb;
                pol(zs::Collapse{(block_cnt + tpb - 1) / tpb, cuda_block_size},
                    [spgv = zs::proxy<space>(spg), sor, clr, rho, dx, ts_c = zs::wrapv<bucket_size>{},
                     tpb_c = zs::wrapv<tpb>{}, pOffset = spg.getPropertyOffset("p"),
                     cutOffset = spg.getPropertyOffset("cut"),
                     blockCnt = block_cnt] __device__(int bid, int tid) mutable {
                        // load halo
                        using vec3i = zs::vec<int, 3>;
                        using spg_t = RM_CVREF_T(spgv);
                        constexpr int tile_size = decltype(ts_c)::value;

                        auto tile = zs::cg::tiled_partition<tile_size>(zs::cg::this_thread_block());
                        auto blockno = bid * RM_CVREF_T(tpb_c)::value + tid / tile_size;
                        if (blockno >= blockCnt)
                            return;

                        auto bcoord = spgv._table._activeKeys[blockno];

                        if (((bcoord[0] & 1) ^ (bcoord[1] & 1) ^ (bcoord[2] & 1)) == clr)
                            return;

                        /// (0, 0, 0),
                        /// (0, 0, 1), (0, 0, -1),
                        /// (0, 1, 0), (0, -1, 0),
                        /// (1, 0, 0), (-1, 0, 0)
                        auto loadVal = [&spgv, &tile, &bcoord, blockno, pOffset,
                                        cutOffset](int k) -> zs::tuple<value_type, value_type> {
                            auto coord = bcoord;
                            coord[k / 2] += k & 1 ? 1 : -1;
                            int bno = spgv._table.tile_query(tile, coord);
                            /// @note default cut value is 1
                            value_type cutVal = 1;
                            if (bno >= 0) {
                                if (k & 1)
                                    cutVal = spgv(cutOffset + (k / 2), bno, 0);
                                else
                                    cutVal = spgv(cutOffset + (k / 2), blockno, 0);
                                return zs::make_tuple(spgv(pOffset, bno, 0), cutVal);
                            } else {
                                if ((k & 1) == 0)
                                    cutVal = spgv(cutOffset + (k / 2), blockno, 0);
                                return zs::make_tuple((value_type)0, cutVal);
                            }
                        };
                        auto div = spgv.value("tmp", blockno, 0);
                        value_type stclVal = 0, cutVal = 0;
                        const auto lane_id = tile.thread_rank();
                        if (lane_id == 0)
                            stclVal = (1 - sor) * spgv(pOffset, blockno, 0);

                        for (int j = 0; j < 6; ++j) {
                            auto [stcl, cut] = loadVal(j);
                            if (lane_id == j + 1) {
                                stclVal = stcl * cut;
                                cutVal = cut;
                            }
                        }
                        for (int stride = 1; stride <= 4; stride <<= 1) {
                            auto stclTmp = tile.shfl(stclVal, lane_id + stride);
                            auto cutTmp = tile.shfl(cutVal, lane_id + stride);
                            if (lane_id + stride < 7 && lane_id != 0) {
                                stclVal += stclTmp;
                                cutVal += cutTmp;
                            }
                        }

                        auto stclSum = tile.shfl(stclVal, 1);
                        auto cutSum = tile.shfl(cutVal, 1);
                        cutSum = zs::max(cutSum, zs::detail::deduce_numeric_epsilon<float>() * 10);
                        if (lane_id == 0) {
                            spgv(pOffset, blockno, 0) = stclVal + sor * (stclSum * dx - div * rho) / (cutSum * dx);
                        }
                    });
#else
                using value_type = typename RM_CVREF_T(spg)::value_type;
                constexpr int side_length = RM_CVREF_T(spg)::side_length;
                static_assert(side_length == 1, "coarsest level block assumed to have a side length of 1.");
                // 7 * sizeof(float) = 28 Bytes / block
                constexpr int arena_size = 7;
                constexpr std::size_t bucket_size = RM_CVREF_T(spg._table)::bucket_size;
                constexpr std::size_t tpb = 8;
                constexpr std::size_t cuda_block_size = bucket_size * tpb;
                pol.shmem(arena_size * sizeof(value_type) * tpb);
                pol(zs::Collapse{(block_cnt + tpb - 1) / tpb, cuda_block_size},
                    [spgv = zs::proxy<space>(spg), dx, rho, sor, clr, ts_c = zs::wrapv<bucket_size>{},
                     tpb_c = zs::wrapv<tpb>{}, pOffset = spg.getPropertyOffset("p"),
                     blockCnt = block_cnt] __device__(value_type * shmem, int bid, int tid) mutable {
                        // load halo
                        using vec3i = zs::vec<int, 3>;
                        using spg_t = RM_CVREF_T(spgv);
                        constexpr int arena_size = 7;
                        constexpr int tile_size = decltype(ts_c)::value;

                        auto tile = zs::cg::tiled_partition<tile_size>(zs::cg::this_thread_block());
                        auto blockno = bid * RM_CVREF_T(tpb_c)::value + tid / tile_size;
                        if (blockno >= blockCnt)
                            return;

                        shmem += (tid / tile_size) * arena_size;
                        auto bcoord = spgv._table._activeKeys[blockno];

                        if (((bcoord[0] & 1) ^ (bcoord[1] & 1) ^ (bcoord[2] & 1)) == clr)
                            return;

                        /// (0, 0, 0),
                        /// (0, 0, 1), (0, 0, -1),
                        /// (0, 1, 0), (0, -1, 0),
                        /// (1, 0, 0), (-1, 0, 0)
                        shmem[0] = spgv(pOffset, blockno, 0);
                        auto loadToShmem = [shmem, &spgv, &tile, pOffset](int shIdx, const vec3i &coord) {
                            int bno = spgv._table.tile_query(tile, coord);
                            if (tile.thread_rank() == 0) {
                                if (bno >= 0)
                                    shmem[shIdx] = spgv(pOffset, bno, 0);
                                else
                                    shmem[shIdx] = 0;
                            }
                        };
                        loadToShmem(1, bcoord + vec3i{0, 0, 1});
                        loadToShmem(2, bcoord + vec3i{0, 0, -1});
                        loadToShmem(3, bcoord + vec3i{0, 1, 0});
                        loadToShmem(4, bcoord + vec3i{0, -1, 0});
                        loadToShmem(5, bcoord + vec3i{1, 0, 0});
                        loadToShmem(6, bcoord + vec3i{-1, 0, 0});

                        tile.sync();

                        if (tile.thread_rank() == 0) {
                            auto div = spgv.value("tmp", blockno, 0);
                            auto p_this = (1 - sor) * shmem[0] +
                                          sor *
                                              ((shmem[1] + shmem[2] + shmem[3] + shmem[4] + shmem[5] + shmem[6]) -
                                               div * dx * dx * rho) /
                                              6;

                            spgv(pOffset, blockno, 0) = p_this;
                        }
                    });
                pol.shmem(0);
#endif
            }
        }
        pol.sync(true);
    }

    template <int level>
    void coloredSOR(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid, float rho, float sor, int nIter) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->getLevel<level>();
        using value_type = typename RM_CVREF_T(spg)::value_type;
        constexpr int side_length = RM_CVREF_T(spg)::side_length;
        constexpr int arena_size = (side_length + 2) * (side_length + 2) * (side_length + 2);
        auto block_cnt = spg.numBlocks();

        auto dx = spg.voxelSize()[0];

        pol.sync(false);
        for (int iter = 0; iter < nIter; ++iter) {
            for (int clr = 0; clr != 2; ++clr) {
                constexpr std::size_t bucket_size = RM_CVREF_T(spg._table)::bucket_size;
                constexpr std::size_t tpb = 2;
                constexpr std::size_t cuda_block_size = bucket_size * tpb;
                pol.shmem(arena_size * sizeof(value_type) * tpb * 4);
                pol(zs::Collapse{(block_cnt + tpb - 1) / tpb, cuda_block_size},
                    [spgv = zs::proxy<space>(spg), dx, rho, clr, sor, ts_c = zs::wrapv<bucket_size>{},
                     tpb_c = zs::wrapv<tpb>{}, pOffset = spg.getPropertyOffset("p"),
                     cutOffset = spg.getPropertyOffset("cut"),
                     blockCnt = block_cnt] __device__(value_type * shmem, int bid, int tid) mutable {
                        // load halo
                        using vec3i = zs::vec<int, 3>;
                        using spg_t = RM_CVREF_T(spgv);
                        constexpr int side_length = spg_t::side_length;
                        constexpr int side_area = side_length * side_length;
                        constexpr int halo_side_length = side_length + 2;
                        constexpr int arena_size = halo_side_length * halo_side_length * halo_side_length;

                        constexpr int block_size = spg_t::block_size;
                        constexpr int half_block_size = spg_t::block_size / 2;
                        constexpr int tile_size = decltype(ts_c)::value;

                        auto halo_index = [](int i, int j, int k) {
                            return i * (halo_side_length * halo_side_length) + j * halo_side_length + k;
                        };
                        auto tile = zs::cg::tiled_partition<tile_size>(zs::cg::this_thread_block());
                        auto blockno = bid * RM_CVREF_T(tpb_c)::value + tid / tile_size;
                        if (blockno >= blockCnt)
                            return;

                        shmem += (tid / tile_size) * arena_size * 4;
                        auto cut0shmem = shmem + arena_size;
                        auto cut1shmem = cut0shmem + arena_size;
                        auto cut2shmem = cut1shmem + arena_size;
                        auto bcoord = spgv._table._activeKeys[blockno];

#if 0
                            if constexpr (true) {
                                auto baseCoord = bcoord - 1;
                                for (int cid = tile.thread_rank(); cid < halo_block_size; cid += tile_size) {
                                    int v = cid;
                                    int k = v % halo_side_length;
                                    v = (v - k) / halo_side_length;
                                    int j = v % halo_side_length;
                                    v = (v - j) / halo_side_length;
                                    int i = v;
                                    auto coord = baseCoord + vec3i{i, j, k};
                                    shmem[cid] = spgv.value(pOffset, coord);
                                    if (i == halo_side_length - 1)
                                        if (blockno < 1)
                                            printf("block [%d] arena [%d, %d, %d] global [%d, %d, %d] from block [%d] "
                                                   "<%d, "
                                                   "%d, %d> : %f (%f)\n",
                                                   (int)blockno, i, j, k, coord[0], coord[1], coord[2], (int)blockno,
                                                   bcoord[0], bcoord[1], bcoord[2], shmem[cid],
                                                   spgv.value(pOffset, coord));
                                }
                            }
#endif

                        auto block = spgv.block(blockno);
                        for (int cid = tile.thread_rank(); cid < block_size; cid += tile_size) {
                            auto localCoord = spg_t::local_offset_to_coord(cid);
                            auto idx = halo_index(localCoord[0] + 1, localCoord[1] + 1, localCoord[2] + 1);
                            shmem[idx] = block(pOffset, cid);
                            cut0shmem[idx] = block(cutOffset + 0, cid);
                            cut1shmem[idx] = block(cutOffset + 1, cid);
                            cut2shmem[idx] = block(cutOffset + 2, cid);
                        }

                        // back
                        int bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, -side_length});
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 1, j + 1, 0)] =
                                    block(pOffset, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 1}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 1, j + 1, 0)] = 0; // no pressure
                            }
                        // front
                        bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, side_length});
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 1, j + 1, halo_side_length - 1)] =
                                    block(pOffset, spg_t::local_coord_to_offset(vec3i{i, j, 0}));
                                cut2shmem[halo_index(i + 1, j + 1, halo_side_length - 1)] =
                                    block(cutOffset + 2, spg_t::local_coord_to_offset(vec3i{i, j, 0}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 1, j + 1, halo_side_length - 1)] = 0; // no pressure
                                cut2shmem[halo_index(i + 1, j + 1, halo_side_length - 1)] = 1;
                            }

                        // bottom
                        bno = spgv._table.tile_query(tile, bcoord + vec3i{0, -side_length, 0});
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 1, 0, j + 1)] =
                                    block(pOffset, spg_t::local_coord_to_offset(vec3i{i, side_length - 1, j}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 1, 0, j + 1)] = 0; // no pressure
                            }
                        // up
                        bno = spgv._table.tile_query(tile, bcoord + vec3i{0, side_length, 0});
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 1, halo_side_length - 1, j + 1)] =
                                    block(pOffset, spg_t::local_coord_to_offset(vec3i{i, 0, j}));
                                cut1shmem[halo_index(i + 1, halo_side_length - 1, j + 1)] =
                                    block(cutOffset + 1, spg_t::local_coord_to_offset(vec3i{i, 0, j}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 1, halo_side_length - 1, j + 1)] = 0; // no pressure
                                cut1shmem[halo_index(i + 1, halo_side_length - 1, j + 1)] = 1;
                            }

                        // left
                        bno = spgv._table.tile_query(tile, bcoord + vec3i{-side_length, 0, 0});
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(0, i + 1, j + 1)] =
                                    block(pOffset, spg_t::local_coord_to_offset(vec3i{side_length - 1, i, j}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(0, i + 1, j + 1)] = 0; // no pressure
                            }
                        // right
                        bno = spgv._table.tile_query(tile, bcoord + vec3i{side_length, 0, 0});
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(halo_side_length - 1, i + 1, j + 1)] =
                                    block(pOffset, spg_t::local_coord_to_offset(vec3i{0, i, j}));
                                cut0shmem[halo_index(halo_side_length - 1, i + 1, j + 1)] =
                                    block(cutOffset + 0, spg_t::local_coord_to_offset(vec3i{0, i, j}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(halo_side_length - 1, i + 1, j + 1)] = 0; // no pressure
                                cut0shmem[halo_index(halo_side_length - 1, i + 1, j + 1)] = 1;
                            }

                        tile.sync();

                        for (int cno = tile.thread_rank(); cno < half_block_size; cno += tile_size) {
                            auto cellno = (cno << 1) | clr;

                            auto ccoord = spgv.local_offset_to_coord(cellno);
                            float div = spgv.value("tmp", blockno, cellno);

                            float p_x[2], p_y[2], p_z[2], p_self{};

                            ccoord += 1;

#if 0
                                    auto check = [&](const auto offset) {
                                        auto c = ccoord + offset;
                                        auto ic = icoord + offset;
                                        auto arenaVal = shmem[halo_index(c[0], c[1], c[2])];
                                        auto actualVal = spgv.value(pOffset, icoord + offset);
                                        if (zs::abs(arenaVal - actualVal) > zs::limits<value_type>::epsilon() * 10) {
                                            printf(
                                                "color %d: blockno %d, coord [center %d] <%d, %d, %d> (offset <%d, %d, "
                                                "%d>): arena val [%f], actual val [%f]\n",
                                                c_clr, (int)blockno, (int)cellno, c[0] - 1, c[1] - 1, c[2] - 1,
                                                (int)offset[0], (int)offset[1], (int)offset[2],
                                                (float)shmem[halo_index(c[0], c[1], c[2])],
                                                (float)spgv.value(pOffset, ic));
                                        }
                                    };
                                    // if (blockno < 3)
                                    if (true) {
                                        check(vec3i::zeros());
                                        check(vec3i{1, 0, 0});
                                        check(vec3i{0, 1, 0});
                                        check(vec3i{0, 0, 1});
                                        check(vec3i{-1, 0, 0});
                                        check(vec3i{0, -1, 0});
                                        check(vec3i{0, 0, -1});
                                    }
#endif

                            auto idx = halo_index(ccoord[0], ccoord[1], ccoord[2]);
                            p_self = shmem[idx];
                            p_x[0] = shmem[idx - halo_side_length * halo_side_length];
                            p_x[1] = shmem[idx + halo_side_length * halo_side_length];

                            p_y[0] = shmem[idx - halo_side_length];
                            p_y[1] = shmem[idx + halo_side_length];

                            p_z[0] = shmem[idx - 1];
                            p_z[1] = shmem[idx + 1];

                            float cut_x[2], cut_y[2], cut_z[2];
                            auto icoord = spgv.iCoord(blockno, cellno);
                            cut_x[0] = cut0shmem[idx];
                            cut_y[0] = cut1shmem[idx];
                            cut_z[0] = cut2shmem[idx];
                            cut_x[1] = cut0shmem[idx + halo_side_length * halo_side_length];
                            cut_y[1] = cut1shmem[idx + halo_side_length];
                            cut_z[1] = cut2shmem[idx + 1];

                            float cut_sum = cut_x[0] + cut_x[1] + cut_y[0] + cut_y[1] + cut_z[0] + cut_z[1];
                            cut_sum = zs::max(cut_sum, zs::detail::deduce_numeric_epsilon<float>() * 10);

                            p_self = (1.f - sor) * p_self +
                                     sor *
                                         ((p_x[0] * cut_x[0] + p_x[1] * cut_x[1] + p_y[0] * cut_y[0] +
                                           p_y[1] * cut_y[1] + p_z[0] * cut_z[0] + p_z[1] * cut_z[1]) *
                                              dx -
                                          div * rho) /
                                         (cut_sum * dx);

                            // spgv(pOffset, blockno, cellno) = p_self;
                            shmem[idx] = p_self;
                        }

                        tile.sync();

                        //
                        for (int cid = tile.thread_rank(); cid < half_block_size; cid += tile_size) {
                            auto cellno = (cid << 1) | clr;
                            auto localCoord = spg_t::local_offset_to_coord(cellno);
                            block(pOffset, cellno) =
                                shmem[halo_index(localCoord[0] + 1, localCoord[1] + 1, localCoord[2] + 1)];
                        }
                    });
                pol.shmem(0);
            } // switch block color
        }     // sor iteration
        pol.sync(true);
        pol.syncCtx();
    }

    template <int level>
    void residual(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid, float rho) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->getLevel<level>();
        auto block_cnt = spg.numBlocks();

        auto dx = spg.voxelSize()[0];

        // shared memory
        using value_type = typename RM_CVREF_T(spg)::value_type;
        constexpr int side_length = RM_CVREF_T(spg)::side_length;
        constexpr int arena_size = (side_length + 2) * (side_length + 2) * (side_length + 2);
        constexpr std::size_t bucket_size = RM_CVREF_T(spg._table)::bucket_size;
        constexpr std::size_t tpb = 2;
        constexpr std::size_t cuda_block_size = bucket_size * tpb;
        pol.shmem(arena_size * sizeof(value_type) * tpb * 4);

        pol(zs::Collapse{(block_cnt + tpb - 1) / tpb, cuda_block_size},
            [spgv = zs::proxy<space>(spg), dx, rho, ts_c = zs::wrapv<bucket_size>{}, tpb_c = zs::wrapv<tpb>{},
             pOffset = spg.getPropertyOffset("p"), cutOffset = spg.getPropertyOffset("cut"),
             blockCnt = block_cnt] __device__(value_type * shmem, int bid, int tid) mutable {
                // load halo
                using vec3i = zs::vec<int, 3>;
                using spg_t = RM_CVREF_T(spgv);
                constexpr int side_length = spg_t::side_length;
                constexpr int side_area = side_length * side_length;
                constexpr int halo_side_length = side_length + 2;
                constexpr int arena_size = halo_side_length * halo_side_length * halo_side_length;

                constexpr int block_size = spg_t::block_size;
                constexpr int tile_size = decltype(ts_c)::value;

                auto halo_index = [](int i, int j, int k) {
                    return k * (halo_side_length * halo_side_length) + j * halo_side_length + i;
                };

                auto tile = zs::cg::tiled_partition<tile_size>(zs::cg::this_thread_block());
                auto blockno = bid * RM_CVREF_T(tpb_c)::value + tid / tile_size;
                if (blockno >= blockCnt)
                    return;

                shmem += (tid / tile_size) * arena_size * 4;
                auto cut0shmem = shmem + arena_size;
                auto cut1shmem = cut0shmem + arena_size;
                auto cut2shmem = cut1shmem + arena_size;
                auto bcoord = spgv._table._activeKeys[blockno];

                auto block = spgv.block(blockno);
                for (int cid = tile.thread_rank(); cid < block_size; cid += tile_size) {
                    auto localCoord = spg_t::local_offset_to_coord(cid);
                    auto idx = halo_index(localCoord[0] + 1, localCoord[1] + 1, localCoord[2] + 1);
                    shmem[idx] = block(pOffset, cid);
                    cut0shmem[idx] = block(cutOffset + 0, cid);
                    cut1shmem[idx] = block(cutOffset + 1, cid);
                    cut2shmem[idx] = block(cutOffset + 2, cid);
                }

                // back
                int bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, -side_length});
                if (bno >= 0) {
                    auto block = spgv.block(bno);
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i + 1, j + 1, 0)] =
                            block(pOffset, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 1}));
                    }
                } else
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i + 1, j + 1, 0)] = 0; // no pressure
                    }
                // front
                bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, side_length});
                if (bno >= 0) {
                    auto block = spgv.block(bno);
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i + 1, j + 1, halo_side_length - 1)] =
                            block(pOffset, spg_t::local_coord_to_offset(vec3i{i, j, 0}));
                        cut2shmem[halo_index(i + 1, j + 1, halo_side_length - 1)] =
                            block(cutOffset + 2, spg_t::local_coord_to_offset(vec3i{i, j, 0}));
                    }
                } else
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i + 1, j + 1, halo_side_length - 1)] = 0; // no pressure
                        cut2shmem[halo_index(i + 1, j + 1, halo_side_length - 1)] = 1;
                    }

                // bottom
                bno = spgv._table.tile_query(tile, bcoord + vec3i{0, -side_length, 0});
                if (bno >= 0) {
                    auto block = spgv.block(bno);
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i + 1, 0, j + 1)] =
                            block(pOffset, spg_t::local_coord_to_offset(vec3i{i, side_length - 1, j}));
                    }
                } else
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i + 1, 0, j + 1)] = 0; // no pressure
                    }
                // up
                bno = spgv._table.tile_query(tile, bcoord + vec3i{0, side_length, 0});
                if (bno >= 0) {
                    auto block = spgv.block(bno);
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i + 1, halo_side_length - 1, j + 1)] =
                            block(pOffset, spg_t::local_coord_to_offset(vec3i{i, 0, j}));
                        cut1shmem[halo_index(i + 1, halo_side_length - 1, j + 1)] =
                            block(cutOffset + 1, spg_t::local_coord_to_offset(vec3i{i, 0, j}));
                    }
                } else
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i + 1, halo_side_length - 1, j + 1)] = 0; // no pressure
                        cut1shmem[halo_index(i + 1, halo_side_length - 1, j + 1)] = 1;
                    }

                // left
                bno = spgv._table.tile_query(tile, bcoord + vec3i{-side_length, 0, 0});
                if (bno >= 0) {
                    auto block = spgv.block(bno);
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(0, i + 1, j + 1)] =
                            block(pOffset, spg_t::local_coord_to_offset(vec3i{side_length - 1, i, j}));
                    }
                } else
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(0, i + 1, j + 1)] = 0; // no pressure
                    }
                // right
                bno = spgv._table.tile_query(tile, bcoord + vec3i{side_length, 0, 0});
                if (bno >= 0) {
                    auto block = spgv.block(bno);
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(halo_side_length - 1, i + 1, j + 1)] =
                            block(pOffset, spg_t::local_coord_to_offset(vec3i{0, i, j}));
                        cut0shmem[halo_index(halo_side_length - 1, i + 1, j + 1)] =
                            block(cutOffset + 0, spg_t::local_coord_to_offset(vec3i{0, i, j}));
                    }
                } else
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(halo_side_length - 1, i + 1, j + 1)] = 0; // no pressure
                        cut0shmem[halo_index(halo_side_length - 1, i + 1, j + 1)] = 1;
                    }

                tile.sync();

                for (int cellno = tile.thread_rank(); cellno < block_size; cellno += tile_size) {
                    float div = spgv.value("tmp", blockno, cellno);

                    auto ccoord = spgv.local_offset_to_coord(cellno);
                    ccoord += 1;

                    float p_x[2], p_y[2], p_z[2], p_self;
                    p_self = shmem[halo_index(ccoord[0], ccoord[1], ccoord[2])];
                    p_x[0] = shmem[halo_index(ccoord[0] - 1, ccoord[1], ccoord[2])];
                    p_x[1] = shmem[halo_index(ccoord[0] + 1, ccoord[1], ccoord[2])];
                    p_y[0] = shmem[halo_index(ccoord[0], ccoord[1] - 1, ccoord[2])];
                    p_y[1] = shmem[halo_index(ccoord[0], ccoord[1] + 1, ccoord[2])];
                    p_z[0] = shmem[halo_index(ccoord[0], ccoord[1], ccoord[2] - 1)];
                    p_z[1] = shmem[halo_index(ccoord[0], ccoord[1], ccoord[2] + 1)];

                    float cut_x[2], cut_y[2], cut_z[2];
                    cut_x[0] = cut0shmem[halo_index(ccoord[0], ccoord[1], ccoord[2])];
                    cut_y[0] = cut1shmem[halo_index(ccoord[0], ccoord[1], ccoord[2])];
                    cut_z[0] = cut2shmem[halo_index(ccoord[0], ccoord[1], ccoord[2])];
                    cut_x[1] = cut0shmem[halo_index(ccoord[0] + 1, ccoord[1], ccoord[2])];
                    cut_y[1] = cut1shmem[halo_index(ccoord[0], ccoord[1] + 1, ccoord[2])];
                    cut_z[1] = cut2shmem[halo_index(ccoord[0], ccoord[1], ccoord[2] + 1)];

                    float m_residual = div - ((p_x[1] - p_self) * cut_x[1] - (p_self - p_x[0]) * cut_x[0] +
                                              (p_y[1] - p_self) * cut_y[1] - (p_self - p_y[0]) * cut_y[0] +
                                              (p_z[1] - p_self) * cut_z[1] - (p_self - p_z[0]) * cut_z[0]) *
                                                 dx / rho;

                    spgv("tmp", 1, blockno, cellno) = m_residual;
                }
            });
        pol.shmem(0);
    }

    float residual_0(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid, float rho) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->getLevel<0>();
        auto block_cnt = spg.numBlocks();

        auto dx = spg.voxelSize()[0];

        // residual
        zs::Vector<float> res{spg.get_allocator(), 0};
        res.resize(block_cnt);
        res.reset(0);

        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), res = zs::proxy<space>(res), dx, rho, pOffset = spg.getPropertyOffset("p"),
             cutOffset = spg.getPropertyOffset("cut")] __device__(int blockno, int cellno) mutable {
                using vec3i = zs::vec<int, 3>;
                auto icoord = spgv.iCoord(blockno, cellno);

                float div = spgv.value("tmp", blockno, cellno);

                float p_self, p_x[2], p_y[2], p_z[2];

                p_self = spgv.value(pOffset, blockno, cellno);
                p_x[0] = spgv.value(pOffset, icoord + vec3i(-1, 0, 0));
                p_x[1] = spgv.value(pOffset, icoord + vec3i(1, 0, 0));
                p_y[0] = spgv.value(pOffset, icoord + vec3i(0, -1, 0));
                p_y[1] = spgv.value(pOffset, icoord + vec3i(0, 1, 0));
                p_z[0] = spgv.value(pOffset, icoord + vec3i(0, 0, -1));
                p_z[1] = spgv.value(pOffset, icoord + vec3i(0, 0, 1));

                float cut_x[2], cut_y[2], cut_z[2];
                cut_x[0] = spgv.value(cutOffset + 0, blockno, cellno);
                cut_y[0] = spgv.value(cutOffset + 1, blockno, cellno);
                cut_z[0] = spgv.value(cutOffset + 2, blockno, cellno);
                cut_x[1] = spgv.value(cutOffset + 0, icoord + vec3i(1, 0, 0), 1.f);
                cut_y[1] = spgv.value(cutOffset + 1, icoord + vec3i(0, 1, 0), 1.f);
                cut_z[1] = spgv.value(cutOffset + 2, icoord + vec3i(0, 0, 1), 1.f);

                float m_residual =
                    div - ((p_x[1] - p_self) * cut_x[1] - (p_self - p_x[0]) * cut_x[0] + (p_y[1] - p_self) * cut_y[1] -
                           (p_self - p_y[0]) * cut_y[0] + (p_z[1] - p_self) * cut_z[1] - (p_self - p_z[0]) * cut_z[0]) *
                              dx / rho;

                zs::atomic_max(zs::exec_cuda, &res[blockno], zs::abs(m_residual));
            });

        float max_residual = reduce(pol, res, thrust::maximum<float>{});
        return max_residual;
    }

    template <int level>
    void restriction(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg_f = NSGrid->getLevel<level>();
        auto &spg_c = NSGrid->getLevel<level + 1>();

        using value_type = typename RM_CVREF_T(spg_f)::value_type;
        constexpr int arena_size = spg_f.block_size;
        pol.shmem(arena_size * sizeof(value_type));

        pol(zs::Collapse{spg_c.numBlocks(), spg_c.block_size},
            [spgv_c = zs::proxy<space>(spg_c),
             spgv_f = zs::proxy<space>(spg_f)] __device__(value_type * shmem, int blockno, int cellno) mutable {
                using vec3i = zs::vec<int, 3>;

                for (int i = cellno; i < spgv_f.block_size; i += spgv_c.block_size)
                    shmem[i] = spgv_f.value("tmp", 1, blockno, i);

                __syncthreads();

                auto ccoord = spgv_c.local_offset_to_coord(cellno) * 2;

                float res_sum = 0;
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j)
                        for (int k = 0; k < 2; ++k) {
                            auto ccoord_f = ccoord + vec3i{i, j, k};
                            auto cellno_f = spgv_f.local_coord_to_offset(ccoord_f);
                            res_sum += shmem[cellno_f];
                        }
                // res_sum /= 8.f;

                spgv_c("tmp", blockno, cellno) = res_sum;
            });
        pol.shmem(0);
    }

    template <int level>
    void prolongation(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg_f = NSGrid->getLevel<level>();
        auto &spg_c = NSGrid->getLevel<level + 1>();

        using value_type = typename RM_CVREF_T(spg_c)::value_type;
        constexpr int arena_size = spg_c.block_size;
        pol.shmem(arena_size * sizeof(value_type));

        pol(zs::Collapse{spg_f.numBlocks(), spg_f.block_size},
            [spgv_f = zs::proxy<space>(spg_f),
             spgv_c = zs::proxy<space>(spg_c)] __device__(value_type * shmem, int blockno, int cellno) mutable {
                for (int i = cellno; i < spgv_c.block_size; i += spgv_f.block_size)
                    shmem[i] = spgv_c.value("p", blockno, i);

                __syncthreads();

                auto ccoord_f = spgv_f.local_offset_to_coord(cellno);
                auto cellno_c = spgv_c.local_coord_to_offset(ccoord_f / 2);

                spgv_f("p", blockno, cellno) += shmem[cellno_c];
            });
        pol.shmem(0);
    }

    template <int level>
    void multigrid(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid, float rho) {
        if constexpr (level == 3) {
            clearInit<level>(pol, NSGrid);

#if ENABLE_PROFILE
            s_timer.tick();
#endif
            redblackSOR<level>(pol, NSGrid, rho, 1.2f, 100);
#if ENABLE_PROFILE
            s_timer.tock();
            s_coarseTime += s_timer.elapsed();
#endif

#if ENABLE_PROFILE
            s_timer.tick();
#endif
            residual<level>(pol, NSGrid, rho);
#if ENABLE_PROFILE
            s_timer.tock();
            s_coarseResTime += s_timer.elapsed();
#endif
            // printf("MG level %d residual: %e\n", level, res);
        } else {
            if constexpr (level != 0)
                clearInit<level>(pol, NSGrid);

#if ENABLE_PROFILE
            s_timer.tick();
#endif
            coloredSOR<level>(pol, NSGrid, rho, 1.2f, 6 + 3 * level);
#if ENABLE_PROFILE
            s_timer.tock();
            s_smoothTime[0] += s_timer.elapsed();
            s_smoothTime[level + 1] += s_timer.elapsed();
#endif

#if ENABLE_PROFILE
            s_timer.tick();
#endif
            residual<level>(pol, NSGrid, rho);
#if ENABLE_PROFILE
            s_timer.tock();
            s_residualTime[0] += s_timer.elapsed();
            s_residualTime[level + 1] += s_timer.elapsed();
#endif
            // printf("MG level %d residual: %e\n", level, res);

#if ENABLE_PROFILE
            s_timer.tick();
#endif
            restriction<level>(pol, NSGrid);
#if ENABLE_PROFILE
            s_timer.tock();
            s_restrictTime[0] += s_timer.elapsed();
            s_restrictTime[level + 1] += s_timer.elapsed();
#endif

            multigrid<level + 1>(pol, NSGrid, rho);

#if ENABLE_PROFILE
            s_timer.tick();
#endif
            prolongation<level>(pol, NSGrid);
#if ENABLE_PROFILE
            s_timer.tock();
            s_prolongateTime[0] += s_timer.elapsed();
            s_prolongateTime[level + 1] += s_timer.elapsed();
#endif

#if ENABLE_PROFILE
            s_timer.tick();
#endif
            coloredSOR<level>(pol, NSGrid, rho, 1.2f, 6 + 3 * level);
#if ENABLE_PROFILE
            s_timer.tock();
            s_smoothTime[0] += s_timer.elapsed();
            s_smoothTime[level + 1] += s_timer.elapsed();
#endif
        }
        return;
    }

    float rightHandSide(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid, float dt, bool hasDiv) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();
        auto dx = spg.voxelSize()[0];

        // shared memory
        using value_type = typename RM_CVREF_T(spg)::value_type;
        constexpr int side_length = RM_CVREF_T(spg)::side_length;
        constexpr int arena_size = (side_length + 1) * (side_length + 1) * (side_length + 1);
        constexpr std::size_t bucket_size = RM_CVREF_T(spg._table)::bucket_size;
        constexpr std::size_t tpb = 4;
        constexpr std::size_t cuda_block_size = bucket_size * tpb;
        pol.shmem(arena_size * sizeof(value_type) * tpb * 3);

        // maximum rhs
        zs::Vector<float> rhs{spg.get_allocator(), block_cnt};
        rhs.reset(0);

        // velocity divergence (source term)
        pol(zs::Collapse{(block_cnt + tpb - 1) / tpb, cuda_block_size},
            [spgv = zs::proxy<space>(spg), rhs = zs::proxy<space>(rhs), ts_c = zs::wrapv<bucket_size>{},
             tpb_c = zs::wrapv<tpb>{}, dxSqOverDt = dx * dx / dt, dx, dt, hasDiv, blockCnt = block_cnt,
             vOffset = spg.getPropertyOffset(src_tag(NSGrid, "v")),
             cutOffset = spg.getPropertyOffset("cut")] __device__(value_type * shmem, int bid, int tid) mutable {
                // load halo
                using vec3i = zs::vec<int, 3>;
                using spg_t = RM_CVREF_T(spgv);
                constexpr int side_length = spg_t::side_length;
                constexpr int side_area = side_length * side_length;
                constexpr int halo_side_length = side_length + 1;
                constexpr int arena_size = halo_side_length * halo_side_length * halo_side_length;

                constexpr int block_size = spg_t::block_size;
                constexpr int tile_size = decltype(ts_c)::value;

                auto halo_index = [](int i, int j, int k) {
                    return k * (halo_side_length * halo_side_length) + j * halo_side_length + i;
                };

                auto tile = zs::cg::tiled_partition<tile_size>(zs::cg::this_thread_block());
                auto blockno = bid * RM_CVREF_T(tpb_c)::value + tid / tile_size;
                if (blockno >= blockCnt)
                    return;

                shmem += (tid / tile_size) * arena_size * 3;
                auto outputShmem = shmem + arena_size;
                auto cutShmem = outputShmem + arena_size;
                auto bcoord = spgv._table._activeKeys[blockno];

                //-------------------u-------------------
                auto block = spgv.block(blockno);
                for (int cid = tile.thread_rank(); cid < block_size; cid += tile_size) {
                    auto localCoord = spg_t::local_offset_to_coord(cid);
                    shmem[halo_index(localCoord[0], localCoord[1], localCoord[2])] = block(vOffset + 0, cid);
                    cutShmem[halo_index(localCoord[0], localCoord[1], localCoord[2])] = block(cutOffset + 0, cid);
                }

                // right
                int bno = spgv._table.tile_query(tile, bcoord + vec3i{side_length, 0, 0});
                if (bno >= 0) {
                    auto block = spgv.block(bno);
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(halo_side_length - 1, i, j)] =
                            block(vOffset + 0, spg_t::local_coord_to_offset(vec3i{0, i, j}));
                        cutShmem[halo_index(halo_side_length - 1, i, j)] =
                            block(cutOffset + 0, spg_t::local_coord_to_offset(vec3i{0, i, j}));
                    }
                } else
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(halo_side_length - 1, i, j)] = 0; // zero velocity
                        cutShmem[halo_index(halo_side_length - 1, i, j)] = 1;
                    }

                tile.sync();

                for (int cellno = tile.thread_rank(); cellno < block_size; cellno += tile_size) {
                    auto ccoord = spgv.local_offset_to_coord(cellno);
                    auto idx = halo_index(ccoord[0], ccoord[1], ccoord[2]);

                    float u_x[2];
                    u_x[0] = shmem[idx];
                    u_x[1] = shmem[idx + 1];

                    float cut_x[2];
                    cut_x[0] = cutShmem[idx];
                    cut_x[1] = cutShmem[idx + 1];

                    outputShmem[idx] = (u_x[1] * cut_x[1] - u_x[0] * cut_x[0]) * dxSqOverDt;

                    spgv("tmp", 1, blockno, cellno) = cut_x[0] + cut_x[1];
                }

                //-------------------v-------------------
                for (int cid = tile.thread_rank(); cid < block_size; cid += tile_size) {
                    auto localCoord = spg_t::local_offset_to_coord(cid);
                    shmem[halo_index(localCoord[0], localCoord[1], localCoord[2])] = block(vOffset + 1, cid);
                    cutShmem[halo_index(localCoord[0], localCoord[1], localCoord[2])] = block(cutOffset + 1, cid);
                }

                // top
                bno = spgv._table.tile_query(tile, bcoord + vec3i{0, side_length, 0});
                if (bno >= 0) {
                    auto block = spgv.block(bno);
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i, halo_side_length - 1, j)] =
                            block(vOffset + 1, spg_t::local_coord_to_offset(vec3i{i, 0, j}));
                        cutShmem[halo_index(i, halo_side_length - 1, j)] =
                            block(cutOffset + 1, spg_t::local_coord_to_offset(vec3i{i, 0, j}));
                    }
                } else
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i, halo_side_length - 1, j)] = 0; // zero velocity
                        cutShmem[halo_index(i, halo_side_length - 1, j)] = 1;
                    }

                tile.sync();

                for (int cellno = tile.thread_rank(); cellno < block_size; cellno += tile_size) {
                    auto ccoord = spgv.local_offset_to_coord(cellno);
                    auto idx = halo_index(ccoord[0], ccoord[1], ccoord[2]);

                    float u_y[2];
                    u_y[0] = shmem[idx];
                    u_y[1] = shmem[idx + halo_side_length];

                    float cut_y[2];
                    cut_y[0] = cutShmem[idx];
                    cut_y[1] = cutShmem[idx + halo_side_length];

                    outputShmem[idx] += (u_y[1] * cut_y[1] - u_y[0] * cut_y[0]) * dxSqOverDt;

                    spgv("tmp", 1, blockno, cellno) += cut_y[0] + cut_y[1];
                }

                //-------------------w-------------------
                for (int cid = tile.thread_rank(); cid < block_size; cid += tile_size) {
                    auto localCoord = spg_t::local_offset_to_coord(cid);
                    shmem[halo_index(localCoord[0], localCoord[1], localCoord[2])] = block(vOffset + 2, cid);
                    cutShmem[halo_index(localCoord[0], localCoord[1], localCoord[2])] = block(cutOffset + 2, cid);
                }

                // back
                bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, side_length});
                if (bno >= 0) {
                    auto block = spgv.block(bno);
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i, j, halo_side_length - 1)] =
                            block(vOffset + 2, spg_t::local_coord_to_offset(vec3i{i, j, 0}));
                        cutShmem[halo_index(i, j, halo_side_length - 1)] =
                            block(cutOffset + 2, spg_t::local_coord_to_offset(vec3i{i, j, 0}));
                    }
                } else
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i, j, halo_side_length - 1)] = 0; // zero velocity
                        cutShmem[halo_index(i, j, halo_side_length - 1)] = 1;
                    }

                tile.sync();

                for (int cellno = tile.thread_rank(); cellno < block_size; cellno += tile_size) {
                    auto ccoord = spgv.local_offset_to_coord(cellno);
                    auto idx = halo_index(ccoord[0], ccoord[1], ccoord[2]);

                    float u_z[2];
                    u_z[0] = shmem[idx];
                    u_z[1] = shmem[idx + halo_side_length * halo_side_length];

                    float cut_z[2];
                    cut_z[0] = cutShmem[idx];
                    cut_z[1] = cutShmem[idx + halo_side_length * halo_side_length];

                    float div_term = outputShmem[idx];
                    div_term += (u_z[1] * cut_z[1] - u_z[0] * cut_z[0]) * dxSqOverDt;

                    float cut_sum = spgv("tmp", 1, blockno, cellno) + cut_z[0] + cut_z[1];

                    if (hasDiv && cut_sum > 3.f) {
                        div_term -= cut_sum / 6.f * spgv.value("div", blockno, cellno) * dx * dxSqOverDt;
                    }

                    spgv("tmp", blockno, cellno) = div_term;

                    zs::atomic_max(zs::exec_cuda, &rhs[blockno], zs::abs(div_term));
                }
            });
        pol.shmem(0);

        float rhs_max = reduce(pol, rhs, thrust::maximum<float>{});
        return rhs_max;
    }

    void projection(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid, ZenoSparseGrid *SolidSDF, float rho,
                    float dt) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->spg;
        auto &sdf = SolidSDF->spg;
        auto block_cnt = spg.numBlocks();
        auto dx = spg.voxelSize()[0];

        // shared memory
        using value_type = typename RM_CVREF_T(spg)::value_type;
        constexpr int side_length = RM_CVREF_T(spg)::side_length;
        constexpr int arena_size = (side_length + 1) * (side_length + 1) * (side_length + 1);
        constexpr std::size_t bucket_size = RM_CVREF_T(spg._table)::bucket_size;
        constexpr std::size_t tpb = 4;
        constexpr std::size_t cuda_block_size = bucket_size * tpb;
        pol.shmem(arena_size * sizeof(value_type) * tpb);

        pol(zs::Collapse{(block_cnt + tpb - 1) / tpb, cuda_block_size},
            [spgv = zs::proxy<space>(spg), sdfv = zs::proxy<space>(sdf), ts_c = zs::wrapv<bucket_size>{},
             tpb_c = zs::wrapv<tpb>{}, dtOverRhoDx = dt / rho / dx, blockCnt = block_cnt,
             vSrcTag = src_tag(NSGrid, "v"), vDstTag = dst_tag(NSGrid, "v"),
             pOffset = spg.getPropertyOffset("p")] __device__(value_type * shmem, int bid, int tid) mutable {
                // load halo
                using vec3i = zs::vec<int, 3>;
                using spg_t = RM_CVREF_T(spgv);
                constexpr int side_length = spg_t::side_length;
                constexpr int side_area = side_length * side_length;
                constexpr int halo_side_length = side_length + 1;
                constexpr int arena_size = halo_side_length * halo_side_length * halo_side_length;

                constexpr int block_size = spg_t::block_size;
                constexpr int tile_size = decltype(ts_c)::value;

                auto halo_index = [](int i, int j, int k) {
                    return k * (halo_side_length * halo_side_length) + j * halo_side_length + i;
                };

                auto tile = zs::cg::tiled_partition<tile_size>(zs::cg::this_thread_block());
                auto blockno = bid * RM_CVREF_T(tpb_c)::value + tid / tile_size;
                if (blockno >= blockCnt)
                    return;

                shmem += (tid / tile_size) * arena_size;
                auto bcoord = spgv._table._activeKeys[blockno];

                auto block = spgv.block(blockno);
                for (int cid = tile.thread_rank(); cid < block_size; cid += tile_size) {
                    auto localCoord = spg_t::local_offset_to_coord(cid);
                    shmem[halo_index(localCoord[0] + 1, localCoord[1] + 1, localCoord[2] + 1)] = block(pOffset, cid);
                }

                // left
                int bno = spgv._table.tile_query(tile, bcoord + vec3i{-side_length, 0, 0});
                if (bno >= 0) {
                    auto block = spgv.block(bno);
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(0, i + 1, j + 1)] =
                            block(pOffset, spg_t::local_coord_to_offset(vec3i{side_length - 1, i, j}));
                    }
                } else
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(0, i + 1, j + 1)] = 0; // zero pressure
                    }

                // bottom
                bno = spgv._table.tile_query(tile, bcoord + vec3i{0, -side_length, 0});
                if (bno >= 0) {
                    auto block = spgv.block(bno);
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i + 1, 0, j + 1)] =
                            block(pOffset, spg_t::local_coord_to_offset(vec3i{i, side_length - 1, j}));
                    }
                } else
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i + 1, 0, j + 1)] = 0; // zero pressure
                    }

                // front
                bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, -side_length});
                if (bno >= 0) {
                    auto block = spgv.block(bno);
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i + 1, j + 1, 0)] =
                            block(pOffset, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 1}));
                    }
                } else
                    for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                        int i = id / side_length;
                        int j = id % side_length;
                        shmem[halo_index(i + 1, j + 1, 0)] = 0; // zero pressure
                    }

                tile.sync();

                for (int cellno = tile.thread_rank(); cellno < block_size; cellno += tile_size) {
                    auto ccoord = spgv.local_offset_to_coord(cellno);
                    ccoord += 1;

                    int idx = halo_index(ccoord[0], ccoord[1], ccoord[2]);
                    int offset[3] = {1, halo_side_length, halo_side_length * halo_side_length};

                    float p_this = shmem[idx];

                    for (int ch = 0; ch < 3; ++ch) {
                        auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);
                        float sdf_f = sdfv.wSample("sdf", wcoord_face);

                        if (sdf_f > 0) {
                            float u = spgv.value(vSrcTag, ch, blockno, cellno);
                            float p_m = shmem[idx - offset[ch]];

                            u -= (p_this - p_m) * dtOverRhoDx;

                            spgv(vDstTag, ch, blockno, cellno) = u;
                        }
                    }
                }
            });
        pol.shmem(0);

        update_cur(NSGrid, "v");
    }

    template <int level>
    void coarseFaceFrac(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg_f = NSGrid->getLevel<level>();
        auto &spg_c = NSGrid->getLevel<level + 1>();

        using value_type = typename RM_CVREF_T(spg_f)::value_type;
        constexpr int arena_size = spg_f.block_size;
        pol.shmem(arena_size * sizeof(value_type));

        pol(zs::Collapse{spg_c.numBlocks(), spg_c.block_size},
            [spgv_c = zs::proxy<space>(spg_c),
             spgv_f = zs::proxy<space>(spg_f)] __device__(value_type * shmem, int blockno, int cellno) mutable {
                using vec3i = zs::vec<int, 3>;
                auto ccoord = spgv_c.local_offset_to_coord(cellno) * 2;

                // x-face
                for (int i = cellno; i < spgv_f.block_size; i += spgv_c.block_size)
                    shmem[i] = spgv_f.value("cut", 0, blockno, i);

                __syncthreads();

                float cut_sum = 0;
                for (int j = 0; j < 2; ++j)
                    for (int k = 0; k < 2; ++k) {
                        auto ccoord_f = ccoord + vec3i{0, j, k};
                        auto cellno_f = spgv_f.local_coord_to_offset(ccoord_f);
                        cut_sum += shmem[cellno_f];
                    }
                cut_sum *= 0.25f;

                spgv_c("cut", 0, blockno, cellno) = cut_sum;

                __syncthreads();

                // y-face
                for (int i = cellno; i < spgv_f.block_size; i += spgv_c.block_size)
                    shmem[i] = spgv_f.value("cut", 1, blockno, i);

                __syncthreads();

                cut_sum = 0;
                for (int i = 0; i < 2; ++i)
                    for (int k = 0; k < 2; ++k) {
                        auto ccoord_f = ccoord + vec3i{i, 0, k};
                        auto cellno_f = spgv_f.local_coord_to_offset(ccoord_f);
                        cut_sum += shmem[cellno_f];
                    }
                cut_sum *= 0.25f;

                spgv_c("cut", 1, blockno, cellno) = cut_sum;

                __syncthreads();

                // z-face
                for (int i = cellno; i < spgv_f.block_size; i += spgv_c.block_size)
                    shmem[i] = spgv_f.value("cut", 2, blockno, i);

                __syncthreads();

                cut_sum = 0;
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j) {
                        auto ccoord_f = ccoord + vec3i{i, j, 0};
                        auto cellno_f = spgv_f.local_coord_to_offset(ccoord_f);
                        cut_sum += shmem[cellno_f];
                    }
                cut_sum *= 0.25f;

                spgv_c("cut", 2, blockno, cellno) = cut_sum;
            });
        pol.shmem(0);
    }

    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto SolidSDF = get_input<ZenoSparseGrid>("SolidSDF");
        auto rho = get_input2<float>("Density");
        auto dt = get_input2<float>("dt");
        auto tolerance = get_input2<float>("Tolerance");
        auto maxIter = get_input2<int>("MaxIterations");
        auto hasDiv = get_input2<bool>("hasDivergence");

        auto pol = zs::cuda_exec();
#if 1
        auto &spg = NSGrid->spg;
        auto &sdf = SolidSDF->spg;
        auto block_cnt = spg.numBlocks();
        auto dx = spg.voxelSize()[0];
        constexpr auto space = zs::execspace_e::cuda;
#endif

#if ENABLE_PROFILE
        s_rhsTime = 0;
        s_projectTime = 0;
        s_mginitTime = 0;
        s_multigridTime = 0;
        s_totalTime = 0;
        s_timer_p.tick();

        s_timer_m.tick();
#endif

        // create right hand side of Poisson equation
#if 1
        float rhs_max = rightHandSide(pol, NSGrid.get(), dt, hasDiv);
#else
        size_t cell_cnt = block_cnt * spg.block_size;
        zs::Vector<float> res{spg.get_allocator(), count_warps(cell_cnt)};
        res.reset(0);

        pol(zs::range(block_cnt * spg.block_size),
            [spgv = zs::proxy<space>(spg), res = zs::proxy<space>(res), cell_cnt, dx, dt,
             vSrcTag = src_tag(NSGrid, "v")] __device__(int cellno) mutable {
                auto icoord = spgv.iCoord(cellno);

                float u_x[2], u_y[2], u_z[2];
                for (int i = 0; i <= 1; ++i) {
                    u_x[i] = spgv.value(vSrcTag, 0, icoord + zs::vec<int, 3>(i, 0, 0));
                    u_y[i] = spgv.value(vSrcTag, 1, icoord + zs::vec<int, 3>(0, i, 0));
                    u_z[i] = spgv.value(vSrcTag, 2, icoord + zs::vec<int, 3>(0, 0, i));
                }

                float div_term = ((u_x[1] - u_x[0]) + (u_y[1] - u_y[0]) + (u_z[1] - u_z[0])) / dx / dt;

                spgv("tmp", cellno / spgv.block_size, cellno % spgv.block_size) = div_term;

                reduce_max(cellno, cell_cnt, zs::abs(div_term), res[cellno / 32]);
            });
        float rhs_max = reduce(pol, res, thrust::maximum<float>{});
#endif

#if ENABLE_PROFILE
        s_timer_m.tock();
        s_rhsTime += s_timer_m.elapsed();

        s_timer_m.tick();
#endif

        // Multi-grid solver with V-Cycle
        printf("========MultiGrid V-cycle Begin========\n");

        clearInit<0>(pol, NSGrid.get());

        coarseFaceFrac<0>(pol, NSGrid.get());
        coarseFaceFrac<1>(pol, NSGrid.get());
        coarseFaceFrac<2>(pol, NSGrid.get());

#if ENABLE_PROFILE
        s_timer_m.tock();
        s_mginitTime += s_timer_m.elapsed();

        s_smoothTime[0] = s_smoothTime[1] = s_smoothTime[2] = s_smoothTime[3] = 0;
        s_restrictTime[0] = s_restrictTime[1] = s_restrictTime[2] = s_restrictTime[3] = 0;
        s_prolongateTime[0] = s_prolongateTime[1] = s_prolongateTime[2] = s_prolongateTime[3] = 0;
        s_coarseTime = 0;
        s_residualTime[0] = s_residualTime[1] = s_residualTime[2] = s_residualTime[3] = 0;
        s_coarseResTime = 0;

        s_timer_m.tick();
#endif

        for (int iter = 0; iter < maxIter; ++iter) {
            multigrid<0>(pol, NSGrid.get(), rho);

            float res = residual_0(pol, NSGrid.get(), rho);
            res /= rhs_max;
            printf("%dth V-cycle residual: %e\n", iter + 1, res);
            if (res < tolerance)
                break;
        }

        printf("========MultiGrid V-cycle End==========\n");

#if ENABLE_PROFILE
        auto str =
            fmt::format("smooth time: ({}: {}, {}, {}); restrict time: ({}: {}, {}, {}).\nprolongate time: ({}: "
                        "{}, {}, {}), residual time: ({}: {}, {}, {}).\ncoarse time: {}, coarse residual time: {}\n",
                        s_smoothTime[0], s_smoothTime[1], s_smoothTime[2], s_smoothTime[3], s_restrictTime[0],
                        s_restrictTime[1], s_restrictTime[2], s_restrictTime[3], s_prolongateTime[0],
                        s_prolongateTime[1], s_prolongateTime[2], s_prolongateTime[3], s_residualTime[0],
                        s_residualTime[1], s_residualTime[2], s_residualTime[3], s_coarseTime, s_coarseResTime);
        zeno::log_warn(str);
        ZS_WARN(str);

        s_timer_m.tock();
        s_multigridTime += s_timer_m.elapsed();

        s_timer_m.tick();
#endif

#if 0
        // pressure projection
        projection(pol, NSGrid.get(), SolidSDF.get(), rho, dt);
#else
        // pressure projection
        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), sdfv = zs::proxy<space>(sdf), dx, dt, rho, vSrcTag = src_tag(NSGrid, "v"),
             vDstTag = dst_tag(NSGrid, "v"),
             pOffset = spg.getPropertyOffset("p")] __device__(int blockno, int cellno) mutable {
                auto icoord = spgv.iCoord(blockno, cellno);
                float p_this = spgv.value(pOffset, blockno, cellno);

                for (int ch = 0; ch < 3; ++ch) {
                    auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);
                    float sdf_f = sdfv.wSample("sdf", wcoord_face);

                    if (sdf_f > 0) {
                        float u = spgv.value(vSrcTag, ch, blockno, cellno);

                        zs::vec<int, 3> offset{0, 0, 0};
                        offset[ch] = -1;

                        float p_m = spgv.value(pOffset, icoord + offset);

                        u -= (p_this - p_m) / dx * dt / rho;

                        spgv(vDstTag, ch, blockno, cellno) = u;
                    }
                }
            });
        update_cur(NSGrid, "v");
#endif

#if ENABLE_PROFILE
        s_timer_m.tock();
        s_projectTime += s_timer_m.elapsed();

        s_timer_p.tock();
        s_totalTime += s_timer_p.elapsed();

        auto str1 = fmt::format(
            "total projection time: {};\nrhs time: {}, mg init time: {}, multigrid solver time: {}, project time: {}\n",
            s_totalTime, s_rhsTime, s_mginitTime, s_multigridTime, s_projectTime);
        zeno::log_warn(str1);
        ZS_WARN(str1);
#endif

        set_output("NSGrid", NSGrid);
    }
};

template <>
void ZSNSPressureProject::coloredSOR<0>(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid, float rho, float sor,
                                        int nIter) {
    constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

    auto &spg = NSGrid->getLevel<0>();
    auto block_cnt = spg.numBlocks();
    auto dx = spg.voxelSize()[0];

    pol.sync(false);
    for (int iter = 0; iter < nIter; ++iter) {
        for (int clr = 0; clr != 2; ++clr) {
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, rho, sor, clr, pOffset = spg.getPropertyOffset("p"),
                 cutOffset = spg.getPropertyOffset("cut")] __device__(int blockno, int cellno) mutable {
                    using vec3i = zs::vec<int, 3>;
                    auto icoord = spgv.iCoord(blockno, cellno);

                    if (((icoord[0] + icoord[1] + icoord[2]) & 1) == clr) {
                        float div = spgv.value("tmp", blockno, cellno);

                        float p_self, p_x[2], p_y[2], p_z[2];

                        p_self = spgv.value(pOffset, blockno, cellno);
                        p_x[0] = spgv.value(pOffset, icoord + vec3i(-1, 0, 0));
                        p_x[1] = spgv.value(pOffset, icoord + vec3i(1, 0, 0));
                        p_y[0] = spgv.value(pOffset, icoord + vec3i(0, -1, 0));
                        p_y[1] = spgv.value(pOffset, icoord + vec3i(0, 1, 0));
                        p_z[0] = spgv.value(pOffset, icoord + vec3i(0, 0, -1));
                        p_z[1] = spgv.value(pOffset, icoord + vec3i(0, 0, 1));

                        float cut_x[2], cut_y[2], cut_z[2];
                        cut_x[0] = spgv.value(cutOffset + 0, blockno, cellno);
                        cut_y[0] = spgv.value(cutOffset + 1, blockno, cellno);
                        cut_z[0] = spgv.value(cutOffset + 2, blockno, cellno);
                        cut_x[1] = spgv.value(cutOffset + 0, icoord + vec3i(1, 0, 0), 1.f);
                        cut_y[1] = spgv.value(cutOffset + 1, icoord + vec3i(0, 1, 0), 1.f);
                        cut_z[1] = spgv.value(cutOffset + 2, icoord + vec3i(0, 0, 1), 1.f);

                        float cut_sum = cut_x[0] + cut_x[1] + cut_y[0] + cut_y[1] + cut_z[0] + cut_z[1];
                        cut_sum = zs::max(cut_sum, zs::detail::deduce_numeric_epsilon<float>() * 10);

                        p_self =
                            (1.f - sor) * p_self + sor *
                                                       ((p_x[0] * cut_x[0] + p_x[1] * cut_x[1] + p_y[0] * cut_y[0] +
                                                         p_y[1] * cut_y[1] + p_z[0] * cut_z[0] + p_z[1] * cut_z[1]) *
                                                            dx -
                                                        div * rho) /
                                                       (cut_sum * dx);

                        spgv(pOffset, blockno, cellno) = p_self;
                    }
                });
        } // switch block color
    }     // sor iteration
    pol.sync(true);
    pol.syncCtx();
}

ZENDEFNODE(ZSNSPressureProject, {/* inputs: */
                                 {"NSGrid",
                                  "SolidSDF",
                                  "dt",
                                  {"float", "Density", "1.0"},
                                  {"float", "Tolerance", "1e-4"},
                                  {"int", "MaxIterations", "10"},
                                  {"bool", "hasDivergence", "0"}},
                                 /* outputs: */
                                 {"NSGrid"},
                                 /* params: */
                                 {},
                                 /* category: */
                                 {"Eulerian"}});

} // namespace zeno