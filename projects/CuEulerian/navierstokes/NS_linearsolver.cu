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

#define ENABLE_PROFILE 1

struct ZSNSPressureProject : INode {

#if ENABLE_PROFILE
    inline static float s_smoothTime[4];
    inline static float s_restrictTime[4];
    inline static float s_prolongateTime[4];
    inline static float s_coarseTime;
    inline static zs::CppTimer s_timer;
    inline static int s_times = 4;
#endif

    template <int level> void clearInit(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->getLevel<level>();
        auto block_cnt = spg.numBlocks();

        // take zero as initial guess
        pol(zs::range(block_cnt * spg.block_size), [spgv = zs::proxy<space>(spg)] __device__(int cellno) mutable {
            auto icoord = spgv.iCoord(cellno);
            spgv("p0", icoord) = 0.f;
        });
    }

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

    template <int level>
    void redblackSOR(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid, float rho, float sor, int nIter) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->getLevel<level>();
        auto block_cnt = spg.numBlocks();

        auto dx = spg.voxelSize()[0];

        for (int iter = 0; iter < nIter; ++iter) {
            // a simple implementation of red-black SOR
            for (int clr = 0; clr != 2; ++clr) {
#if 0
                pol(zs::range(block_cnt * spg.block_size), [spgv = zs::proxy<space>(spg), dx, rho, sor,
                                                            clr] __device__(int cellno) mutable {
                    auto icoord = spgv.iCoord(cellno);

                    if (((icoord[0] + icoord[1] + icoord[2]) & 1) == clr) {
                        float div = spgv.value("tmp", icoord);

                        const int stcl = 1; // stencil point in each side
                        float p_x[2 * stcl + 1], p_y[2 * stcl + 1], p_z[2 * stcl + 1];

                        for (int i = -stcl; i <= stcl; ++i) {
                            p_x[i + stcl] = spgv.value("p0", icoord + zs::vec<int, 3>(i, 0, 0));
                            p_y[i + stcl] = spgv.value("p0", icoord + zs::vec<int, 3>(0, i, 0));
                            p_z[i + stcl] = spgv.value("p0", icoord + zs::vec<int, 3>(0, 0, i));
                        }

                        float p_this =
                            (1.f - sor) * p_x[stcl] +
                            sor * ((p_x[0] + p_x[2] + p_y[0] + p_y[2] + p_z[0] + p_z[2]) - div * dx * dx * rho) / 6.f;

                        spgv("p0", icoord) = p_this;
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
                     tpb_c = zs::wrapv<tpb>{}, pOffset = spg.getPropertyOffset("p0"),
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

                        __threadfence();
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

        for (int iter = 0; iter < nIter; ++iter) {
            for (int clr = 0; clr != 2; ++clr) {
                if (false) {
                    constexpr int tile_size = zs::math::min((int)RM_CVREF_T(spg)::block_size, (int)32);
                    pol(zs::range(block_cnt * tile_size), [spgv = zs::proxy<space>(spg), dx, rho, clr, sor,
                                                           depth = 3 - level,
                                                           ts_c = zs::wrapv<tile_size>{}] __device__(int tid) mutable {
                        constexpr int side_length = RM_CVREF_T(spgv)::side_length;
                        constexpr int half_block_size = RM_CVREF_T(spgv)::block_size / 2;
                        constexpr int tile_size = decltype(ts_c)::value;
                        auto tile = zs::cg::tiled_partition<tile_size>(zs::cg::this_thread_block());
                        auto blockno = tid / tile_size;

                        auto bcoord = spgv._table._activeKeys[blockno];
                        if ((((bcoord[0] & side_length) ^ (bcoord[1] & side_length) ^ (bcoord[2] & side_length)) >>
                             depth) == clr)
                            return;

                        for (int c_clr = 0; c_clr != 2; ++c_clr) {
                            for (int cno = tile.thread_rank(); cno < half_block_size; cno += tile_size) {
                                auto cellno = (cno << 1) | c_clr;

                                auto ccoord = spgv.local_offset_to_coord(cellno);
                                auto icoord = bcoord + ccoord;

                                float div = spgv.value("tmp", blockno, cellno);

                                const int stcl = 1; // stencil point in each side
                                float p_x[2 * stcl + 1], p_y[2 * stcl + 1], p_z[2 * stcl + 1];

                                for (int i = -stcl; i <= stcl; ++i) {
                                    p_x[i + stcl] = spgv.value("p0", icoord + zs::vec<int, 3>(i, 0, 0));
                                    p_y[i + stcl] = spgv.value("p0", icoord + zs::vec<int, 3>(0, i, 0));
                                    p_z[i + stcl] = spgv.value("p0", icoord + zs::vec<int, 3>(0, 0, i));
                                }

                                float p_this =
                                    (1.f - sor) * p_x[stcl] +
                                    sor *
                                        ((p_x[0] + p_x[2] + p_y[0] + p_y[2] + p_z[0] + p_z[2]) - div * dx * dx * rho) /
                                        6.f;

                                spgv("p0", blockno, cellno) = p_this;
                            }
                        }
                    });
                } else {
                    constexpr std::size_t bucket_size = RM_CVREF_T(spg._table)::bucket_size;
                    constexpr std::size_t tpb = 2;
                    constexpr std::size_t cuda_block_size = bucket_size * tpb;
                    pol.shmem(arena_size * sizeof(value_type) * tpb);
                    pol(zs::Collapse{(block_cnt + tpb - 1) / tpb, cuda_block_size},
                        [spgv = zs::proxy<space>(spg), dx, rho, clr, sor, ts_c = zs::wrapv<bucket_size>{},
                         tpb_c = zs::wrapv<tpb>{}, pOffset = spg.getPropertyOffset("p0"), blockCnt = block_cnt,
                         depth = 3 - level] __device__(value_type * shmem, int bid, int tid) mutable {
                            // load halo
                            using vec3i = zs::vec<int, 3>;
                            using spg_t = RM_CVREF_T(spgv);
                            constexpr int side_length = spg_t::side_length;
                            constexpr int side_area = side_length * side_length;
                            constexpr int halo_side_length = side_length + 2;
                            constexpr int halo_block_size = halo_side_length * halo_side_length * halo_side_length;

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

                            shmem += (tid / tile_size) * arena_size;
                            auto bcoord = spgv._table._activeKeys[blockno];

                            if ((((bcoord[0] & side_length) ^ (bcoord[1] & side_length) ^ (bcoord[2] & side_length)) >>
                                 depth) == clr)
                                return;

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
                                shmem[halo_index(localCoord[0] + 1, localCoord[1] + 1, localCoord[2] + 1)] =
                                    block(pOffset, cid);
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
                                }
                            } else
                                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                    int i = id / side_length;
                                    int j = id % side_length;
                                    shmem[halo_index(i + 1, j + 1, halo_side_length - 1)] = 0; // no pressure
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
                                }
                            } else
                                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                    int i = id / side_length;
                                    int j = id % side_length;
                                    shmem[halo_index(i + 1, halo_side_length - 1, j + 1)] = 0; // no pressure
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
                                }
                            } else
                                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                    int i = id / side_length;
                                    int j = id % side_length;
                                    shmem[halo_index(halo_side_length - 1, i + 1, j + 1)] = 0; // no pressure
                                }

                            __threadfence();
                            tile.sync();

                            for (int c_clr = 0; c_clr != 2; ++c_clr) {

                                for (int cno = tile.thread_rank(); cno < half_block_size; cno += tile_size) {
                                    auto cellno = (cno << 1) | c_clr;

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

                                    float p_this = (1.f - sor) * p_self +
                                                   sor *
                                                       ((p_x[0] + p_x[1] + p_y[0] + p_y[1] + p_z[0] + p_z[1]) -
                                                        div * dx * dx * rho) /
                                                       6.f;

                                    // spgv(pOffset, blockno, cellno) = p_this;
                                    shmem[idx] = p_this;
                                }

                                __threadfence();
                                // tile.sync();
                            }

                            //
                            for (int cid = tile.thread_rank(); cid < block_size; cid += tile_size) {
                                auto localCoord = spg_t::local_offset_to_coord(cid);
                                block(pOffset, cid) =
                                    shmem[halo_index(localCoord[0] + 1, localCoord[1] + 1, localCoord[2] + 1)];
                            }
                        });
                    pol.shmem(0);
                }
            } // switch block color
        }     // sor iteration
    }

    template <int level> float residual(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid, float rho) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->getLevel<level>();
        auto block_cnt = spg.numBlocks();

        auto dx = spg.voxelSize()[0];

        int cur = level == 0 ? NSGrid->readMeta<int>("p_cur") : 0;

        // residual
        size_t cell_cnt = block_cnt * spg.block_size;
        zs::Vector<float> res{spg.get_allocator(), count_warps(cell_cnt)};
        res.reset(0);

        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), res = zs::proxy<space>(res), cell_cnt, dx, rho,
             pSrcTag = zs::SmallString{std::string("p") + std::to_string(cur)}] __device__(int blockno,
                                                                                           int cellno) mutable {
                auto icoord = spgv.iCoord(blockno, cellno);

                float div = spgv.value("tmp", icoord);

                const int stcl = 1; // stencil point in each side
                float p_x[2 * stcl + 1], p_y[2 * stcl + 1], p_z[2 * stcl + 1];

                for (int i = -stcl; i <= stcl; ++i) {
                    p_x[i + stcl] = spgv.value(pSrcTag, icoord + zs::vec<int, 3>(i, 0, 0));
                    p_y[i + stcl] = spgv.value(pSrcTag, icoord + zs::vec<int, 3>(0, i, 0));
                    p_z[i + stcl] = spgv.value(pSrcTag, icoord + zs::vec<int, 3>(0, 0, i));
                }

                float m_residual = div - (scheme::central_diff_2nd(p_x[0], p_x[1], p_x[2], dx) +
                                          scheme::central_diff_2nd(p_y[0], p_y[1], p_y[2], dx) +
                                          scheme::central_diff_2nd(p_z[0], p_z[1], p_z[2], dx)) /
                                             rho;

                spgv("tmp", 1, blockno, cellno) = m_residual;

                size_t cellno_glb = blockno * spgv.block_size + cellno;

                reduce_max(cellno_glb, cell_cnt, zs::abs(m_residual), res[cellno_glb / 32]);
            });

        float max_residual = reduce(pol, res, thrust::maximum<float>{});
        return max_residual;
    }

    template <int level> void restriction(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg_f = NSGrid->getLevel<level>();
        auto &spg_c = NSGrid->getLevel<level + 1>();

        pol(zs::Collapse{spg_c.numBlocks(), spg_c.block_size},
            [spgv_c = zs::proxy<space>(spg_c), spgv_f = zs::proxy<space>(spg_f)] __device__(int blockno,
                                                                                            int cellno) mutable {
                auto bcoord = spgv_c._table._activeKeys[blockno];
                auto ccoord = spgv_c.local_offset_to_coord(cellno);

                auto icoord_c = bcoord + ccoord;
                auto icoord_f = bcoord * 2 + ccoord * 2;

                float res_sum = 0;
                for (int k = 0; k < 2; ++k)
                    for (int j = 0; j < 2; ++j)
                        for (int i = 0; i < 2; ++i) {
                            res_sum += spgv_f.value("tmp", 1, icoord_f + zs::vec<int, 3>(i, j, k));
                        }

                spgv_c("tmp", icoord_c) = res_sum / 8.f;
            });
    }

    template <int level> void prolongation(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid) {
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg_f = NSGrid->getLevel<level>();
        auto &spg_c = NSGrid->getLevel<level + 1>();

        int cur = level == 0 ? NSGrid->readMeta<int>("p_cur") : 0;

        pol(zs::Collapse{spg_f.numBlocks(), spg_f.block_size},
            [spgv_f = zs::proxy<space>(spg_f), spgv_c = zs::proxy<space>(spg_c),
             pSrcTag = zs::SmallString{std::string("p") + std::to_string(cur)}] __device__(int blockno,
                                                                                           int cellno) mutable {
                auto bcoord = spgv_f._table._activeKeys[blockno];
                auto ccoord = spgv_f.local_offset_to_coord(cellno);

                auto icoord_f = bcoord + ccoord;
                auto icoord_c = bcoord / 2 + ccoord / 2;

                spgv_f(pSrcTag, icoord_f) += spgv_c("p0", icoord_c);
            });
    }

    template <int level> void multigrid(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid, float rho) {
        if constexpr (level == 3) {
            clearInit<level>(pol, NSGrid);
            //Jacobi<level>(pol, NSGrid, rho, 100);

#if ENABLE_PROFILE
            s_timer.tick();
#endif
            redblackSOR<level>(pol, NSGrid, rho, 1.2f, 100);
#if ENABLE_PROFILE
            s_timer.tock();
            s_coarseTime += s_timer.elapsed();
#endif

            float res = residual<level>(pol, NSGrid, rho);
            printf("MG level %d residual: %e\n", level, res);
        } else {
            if constexpr (level != 0)
                clearInit<level>(pol, NSGrid);

#if ENABLE_PROFILE
            s_timer.tick();
#endif
            coloredSOR<level>(pol, NSGrid, rho, 1.2f, 4);
#if ENABLE_PROFILE
            s_timer.tock();
            s_smoothTime[0] += s_timer.elapsed();
            s_smoothTime[level + 1] += s_timer.elapsed();
#endif

            float res = residual<level>(pol, NSGrid, rho);
            printf("MG level %d residual: %e\n", level, res);

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
            coloredSOR<level>(pol, NSGrid, rho, 1.2f, 4);
#if ENABLE_PROFILE
            s_timer.tock();
            s_smoothTime[0] += s_timer.elapsed();
            s_smoothTime[level + 1] += s_timer.elapsed();
#endif
        }
        return;
    }

    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto rho = get_input2<float>("Density");
        auto dt = get_input2<float>("dt");
        auto maxIter = get_input2<int>("MaxIterations");

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();
        auto dx = spg.voxelSize()[0];

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        // velocity divergence (source term)
        pol(zs::range(block_cnt * spg.block_size),
            [spgv = zs::proxy<space>(spg), dx, dt, vSrcTag = src_tag(NSGrid, "v")] __device__(int cellno) mutable {
                auto icoord = spgv.iCoord(cellno);

                float u_x[2], u_y[2], u_z[2];
                for (int i = 0; i <= 1; ++i) {
                    u_x[i] = spgv.value(vSrcTag, 0, icoord + zs::vec<int, 3>(i, 0, 0));
                    u_y[i] = spgv.value(vSrcTag, 1, icoord + zs::vec<int, 3>(0, i, 0));
                    u_z[i] = spgv.value(vSrcTag, 2, icoord + zs::vec<int, 3>(0, 0, i));
                }

                float div_term = ((u_x[1] - u_x[0]) + (u_y[1] - u_y[0]) + (u_z[1] - u_z[0])) / dx / dt;

                spgv("tmp", icoord) = div_term;
            });

        // Multi-grid solver with V-Cycle
        const float tolerence = 1e-6;
        printf("========MultiGrid V-cycle Begin========\n");

#if ENABLE_PROFILE
        s_smoothTime[0] = s_smoothTime[1] = s_smoothTime[2] = s_smoothTime[3] = 0;
        s_restrictTime[0] = s_restrictTime[1] = s_restrictTime[2] = s_restrictTime[3] = 0;
        s_prolongateTime[0] = s_prolongateTime[1] = s_prolongateTime[2] = s_prolongateTime[3] = 0;
        s_coarseTime = 0;
#endif

        for (int iter = 0; iter < maxIter; ++iter) {
            printf("-----%dth V-cycle-----\n", iter);
            multigrid<0>(pol, NSGrid.get(), rho);
            float res = residual<0>(pol, NSGrid.get(), rho);
            if (res < tolerence)
                break;
        }

#if ENABLE_PROFILE
        auto str = fmt::format("smooth time: ({}: {}, {}, {}); restrict time: ({}: {}, {}, {}).\nprolongate time: ({}: "
                               "{}, {}, {}), coarse time: {}\n",
                               s_smoothTime[0], s_smoothTime[1], s_smoothTime[2], s_smoothTime[3], s_restrictTime[0],
                               s_restrictTime[1], s_restrictTime[2], s_restrictTime[3], s_prolongateTime[0],
                               s_prolongateTime[1], s_prolongateTime[2], s_prolongateTime[3], s_coarseTime);
        zeno::log_warn(str);
        ZS_WARN(str);
#endif

        printf("========MultiGrid V-cycle End========\n");

        // pressure projection
        pol(zs::range(block_cnt * spg.block_size),
            [spgv = zs::proxy<space>(spg), dx, dt, rho, vSrcTag = src_tag(NSGrid, "v"), vDstTag = dst_tag(NSGrid, "v"),
             pSrcTag = src_tag(NSGrid, "p")] __device__(int cellno) mutable {
                auto icoord = spgv.iCoord(cellno);
                float p_this = spgv.value(pSrcTag, icoord);

                for (int ch = 0; ch < 3; ++ch) {
                    float u = spgv.value(vSrcTag, ch, icoord);

                    zs::vec<int, 3> offset{0, 0, 0};
                    offset[ch] = -1;

                    float p_m = spgv.value(pSrcTag, icoord + offset);

                    u -= (p_this - p_m) / dx * dt / rho;

                    spgv(vDstTag, ch, icoord) = u;
                }
            });
        update_cur(NSGrid, "v");

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSNSPressureProject, {/* inputs: */
                                 {"NSGrid", "dt", {"float", "Density", "1.0"}, {"int", "MaxIterations", "10"}},
                                 /* outputs: */
                                 {"NSGrid"},
                                 /* params: */
                                 {},
                                 /* category: */
                                 {"Eulerian"}});

} // namespace zeno