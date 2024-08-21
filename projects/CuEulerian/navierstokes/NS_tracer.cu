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

#include "../scheme.hpp"
#include "../utils.cuh"
#include "Noise.cuh"

namespace zeno {

struct ZSTracerAdvectDiffuse : INode {
    void compute(zs::CudaExecutionPolicy &pol, zs::SmallString tag, float diffuse, float dt, std::string scheme,
                 float speedScale, zeno::vec3f wind, ZenoSparseGrid *NSGrid) {

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto dx = spg.voxelSize()[0];

        if (scheme == "Semi-Lagrangian") {
            // Semi-Lagrangian advection (1st order)
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, speedScale, wind = zs::vec<float, 3>::from_array(wind),
                 vSrcTag = src_tag(NSGrid, "v"), trcSrcTag = src_tag(NSGrid, tag),
                 trcDstTag = dst_tag(NSGrid, tag)] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    auto wcoord = spgv.indexToWorld(icoord);

                    auto u_adv = (spgv.iStaggeredPack(vSrcTag, icoord) + wind) * speedScale;
                    float trc_sl = spgv.wSample(trcSrcTag, wcoord - u_adv * dt);

                    spgv(trcDstTag, blockno, cellno) = trc_sl;
                });

            update_cur(NSGrid, tag);
        } else if (scheme == "MacCormack") {
            // MacCormack scheme
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, speedScale, wind = zs::vec<float, 3>::from_array(wind),
                 vSrcTag = src_tag(NSGrid, "v"), trcSrcTag = src_tag(NSGrid, tag),
                 trcDstTag = zs::SmallString{"tmp"}] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    auto wcoord = spgv.indexToWorld(icoord);

                    auto u_adv = (spgv.iStaggeredPack(vSrcTag, icoord) + wind) * speedScale;
                    float trc_sl = spgv.wSample(trcSrcTag, wcoord - u_adv * dt);

                    spgv(trcDstTag, blockno, cellno) = trc_sl;
                });
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, speedScale, wind = zs::vec<float, 3>::from_array(wind),
                 vSrcTag = src_tag(NSGrid, "v"), trcTag = src_tag(NSGrid, tag), trcSrcTag = zs::SmallString{"tmp"},
                 trcDstTag = dst_tag(NSGrid, tag)] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    auto wcoord = spgv.indexToWorld(icoord);

                    auto u_adv = (spgv.iStaggeredPack(vSrcTag, icoord) + wind) * speedScale;
                    float trc_sl = spgv.wSample(trcSrcTag, wcoord + u_adv * dt);

                    float trc_mc =
                        spgv.value(trcSrcTag, blockno, cellno) + (spgv.value(trcTag, blockno, cellno) - trc_sl) / 2.f;

                    // clamp
                    auto icoord_src = spgv.worldToIndex(wcoord - u_adv * dt);
                    auto arena = spgv.iArena(icoord_src);
                    auto sl_mi = arena.minimum(trcTag);
                    auto sl_ma = arena.maximum(trcTag);
                    if (trc_mc > sl_ma || trc_mc < sl_mi) {
                        trc_mc = arena.isample(trcTag, 0, spgv._background);
                    }

                    spgv(trcDstTag, blockno, cellno) = trc_mc;
                });

            update_cur(NSGrid, tag);
        } else if (scheme == "BFECC") {
            // Back and Forth Error Compensation and Correction (BFECC)
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, speedScale, wind = zs::vec<float, 3>::from_array(wind),
                 vSrcTag = src_tag(NSGrid, "v"), trcSrcTag = src_tag(NSGrid, tag),
                 trcDstTag = dst_tag(NSGrid, tag)] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    auto wcoord = spgv.indexToWorld(icoord);

                    auto u_adv = (spgv.iStaggeredPack(vSrcTag, icoord) + wind) * speedScale;
                    float trc_sl = spgv.wSample(trcSrcTag, wcoord - u_adv * dt);

                    spgv(trcDstTag, blockno, cellno) = trc_sl;
                });
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, speedScale, wind = zs::vec<float, 3>::from_array(wind),
                 vSrcTag = src_tag(NSGrid, "v"), trcTag = src_tag(NSGrid, tag), trcSrcTag = dst_tag(NSGrid, tag),
                 trcDstTag = zs::SmallString{"tmp"}] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    auto wcoord = spgv.indexToWorld(icoord);

                    auto u_adv = (spgv.iStaggeredPack(vSrcTag, icoord) + wind) * speedScale;
                    float trc_sl = spgv.wSample(trcSrcTag, wcoord + u_adv * dt);
                    float trc_n = spgv.value(trcTag, blockno, cellno);

                    spgv(trcDstTag, blockno, cellno) = trc_n + (trc_n - trc_sl) / 2.f;
                });
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, speedScale, wind = zs::vec<float, 3>::from_array(wind),
                 vSrcTag = src_tag(NSGrid, "v"), trcTag = src_tag(NSGrid, tag), trcSrcTag = zs::SmallString{"tmp"},
                 trcDstTag = dst_tag(NSGrid, tag)] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    auto wcoord = spgv.indexToWorld(icoord);

                    auto u_adv = (spgv.iStaggeredPack(vSrcTag, icoord) + wind) * speedScale;
                    auto icoord_src = spgv.worldToIndex(wcoord - u_adv * dt);
                    auto arena = spgv.iArena(icoord_src);

                    float trc_sl = arena.isample(trcSrcTag, 0, spgv._background);

                    // clamp
                    auto sl_mi = arena.minimum(trcTag);
                    auto sl_ma = arena.maximum(trcTag);
                    if (trc_sl > sl_ma || trc_sl < sl_mi) {
                        trc_sl = arena.isample(trcTag, 0, spgv._background);
                    }

                    spgv(trcDstTag, blockno, cellno) = trc_sl;
                });

            update_cur(NSGrid, tag);
        } else if (scheme == "Stencil") {
            {
                // shared memory
                using value_type = typename RM_CVREF_T(spg)::value_type;
                constexpr int side_length = RM_CVREF_T(spg)::side_length;
                constexpr int arena_size = (side_length + 3) * (side_length + 3) * (side_length + 3);
                constexpr std::size_t bucket_size = RM_CVREF_T(spg._table)::bucket_size;
                constexpr std::size_t tpb = 4;
                constexpr std::size_t cuda_block_size = bucket_size * tpb;
                pol.shmem(arena_size * sizeof(value_type) * tpb);

                // Finite Volume Method (FVM)
                // numrtical flux of tracer
                pol(zs::Collapse{(block_cnt + tpb - 1) / tpb, cuda_block_size},
                    [spgv = zs::proxy<space>(spg), dx, speedScale, wind = zs::vec<float, 3>::from_array(wind),
                     ts_c = zs::wrapv<bucket_size>{}, tpb_c = zs::wrapv<tpb>{}, blockCnt = block_cnt,
                     tagOffset = spg.getPropertyOffset(src_tag(NSGrid, tag)),
                     vSrcOffset = spg.getPropertyOffset(src_tag(NSGrid, "v"))] __device__(value_type * shmem, int bid,
                                                                                          int tid) mutable {
                        using vec3i = zs::vec<int, 3>;
                        using spg_t = RM_CVREF_T(spgv);
                        constexpr int side_length = spg_t::side_length;
                        constexpr int side_area = side_length * side_length;
                        constexpr int halo_side_length = side_length + 3;
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

                        // load halo
                        for (int cid = tile.thread_rank(); cid < block_size; cid += tile_size) {
                            auto localCoord = spg_t::local_offset_to_coord(cid);
                            auto idx = halo_index(localCoord[0] + 2, localCoord[1] + 2, localCoord[2] + 2);
                            shmem[idx] = block(tagOffset, cid);
                        }

                        // back
                        int bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, -side_length});
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 2, j + 2, 0)] =
                                    block(tagOffset, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 2}));
                                shmem[halo_index(i + 2, j + 2, 1)] =
                                    block(tagOffset, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 1}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 2, j + 2, 0)] = 0;
                                shmem[halo_index(i + 2, j + 2, 1)] = 0;
                            }
                        // front
                        bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, side_length});
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 2, j + 2, halo_side_length - 1)] =
                                    block(tagOffset, spg_t::local_coord_to_offset(vec3i{i, j, 0}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 2, j + 2, halo_side_length - 1)] = 0;
                            }

                        // bottom
                        bno = spgv._table.tile_query(tile, bcoord + vec3i{0, -side_length, 0});
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 2, 0, j + 2)] =
                                    block(tagOffset, spg_t::local_coord_to_offset(vec3i{i, side_length - 2, j}));
                                shmem[halo_index(i + 2, 1, j + 2)] =
                                    block(tagOffset, spg_t::local_coord_to_offset(vec3i{i, side_length - 1, j}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 2, 0, j + 2)] = 0;
                                shmem[halo_index(i + 2, 1, j + 2)] = 0;
                            }
                        // up
                        bno = spgv._table.tile_query(tile, bcoord + vec3i{0, side_length, 0});
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 2, halo_side_length - 1, j + 2)] =
                                    block(tagOffset, spg_t::local_coord_to_offset(vec3i{i, 0, j}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 2, halo_side_length - 1, j + 2)] = 0;
                            }

                        // left
                        bno = spgv._table.tile_query(tile, bcoord + vec3i{-side_length, 0, 0});
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(0, i + 2, j + 2)] =
                                    block(tagOffset, spg_t::local_coord_to_offset(vec3i{side_length - 2, i, j}));
                                shmem[halo_index(1, i + 2, j + 2)] =
                                    block(tagOffset, spg_t::local_coord_to_offset(vec3i{side_length - 1, i, j}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(0, i + 2, j + 2)] = 0;
                                shmem[halo_index(1, i + 2, j + 2)] = 0;
                            }
                        // right
                        bno = spgv._table.tile_query(tile, bcoord + vec3i{side_length, 0, 0});
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(halo_side_length - 1, i + 2, j + 2)] =
                                    block(tagOffset, spg_t::local_coord_to_offset(vec3i{0, i, j}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(halo_side_length - 1, i + 2, j + 2)] = 0;
                            }

                        tile.sync();

                        for (int cellno = tile.thread_rank(); cellno < block_size; cellno += tile_size) {
                            auto ccoord = spgv.local_offset_to_coord(cellno);
                            ccoord += 2;

                            const int stcl = 2; // stencil point in each side
                            float trc[3][2 * stcl];

                            // | i - 2 | i - 1 | i | i + 1 |
                            for (int i = -stcl; i < stcl; ++i) {
                                trc[0][i + stcl] = shmem[halo_index(ccoord[0] + i, ccoord[1], ccoord[2])];
                                trc[1][i + stcl] = shmem[halo_index(ccoord[0], ccoord[1] + i, ccoord[2])];
                                trc[2][i + stcl] = shmem[halo_index(ccoord[0], ccoord[1], ccoord[2] + i)];
                            }

                            float u_adv[3];
                            for (int ch = 0; ch < 3; ++ch)
                                u_adv[ch] = (block(vSrcOffset + ch, cellno) + wind[ch]) * speedScale;

                            // approximate value at i - 1/2
                            float flux[3];
                            for (int ch = 0; ch < 3; ++ch) {
                                // convection flux
                                if (u_adv[ch] < 0)
                                    flux[ch] = u_adv[ch] * scheme::TVD_MUSCL3(trc[ch][1], trc[ch][2], trc[ch][3]);
                                else
                                    flux[ch] = u_adv[ch] * scheme::TVD_MUSCL3(trc[ch][2], trc[ch][1], trc[ch][0]);
                            }

                            for (int ch = 0; ch < 3; ++ch) {
                                spgv("tmp", ch, blockno, cellno) = flux[ch];
                            }
                        }
                    });
                pol.shmem(0);
            }

            // time integration of tracer
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, tag = src_tag(NSGrid, tag)] __device__(int blockno,
                                                                                              int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);

                    float flux[3][2];
                    for (int ch = 0; ch < 3; ++ch) {
                        zs::vec<int, 3> offset{0, 0, 0};
                        offset[ch] = 1;

                        flux[ch][0] = spgv.value("tmp", ch, blockno, cellno);
                        flux[ch][1] = spgv.value("tmp", ch, icoord + offset);
                    }

                    float dtrc = 0;
                    for (int ch = 0; ch < 3; ++ch) {
                        dtrc += (flux[ch][0] - flux[ch][1]) / dx;
                    }
                    dtrc *= dt;

                    spgv(tag, blockno, cellno) += dtrc;
                });
        } else {
            throw std::runtime_error(fmt::format("Advection scheme [{}] not found!", scheme));
        }

        if (diffuse > 0) {
            // shared memory
            using value_type = typename RM_CVREF_T(spg)::value_type;
            constexpr int side_length = RM_CVREF_T(spg)::side_length;
            constexpr int arena_size = (side_length + 2) * (side_length + 2) * (side_length + 2);
            constexpr std::size_t bucket_size = RM_CVREF_T(spg._table)::bucket_size;
            constexpr std::size_t tpb = 4;
            constexpr std::size_t cuda_block_size = bucket_size * tpb;
            pol.shmem(arena_size * sizeof(value_type) * tpb);

            // diffusion
            pol(zs::Collapse{(block_cnt + tpb - 1) / tpb, cuda_block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, diffuse, ts_c = zs::wrapv<bucket_size>{},
                 tpb_c = zs::wrapv<tpb>{}, blockCnt = block_cnt,
                 trcSrcOffset = spg.getPropertyOffset(src_tag(NSGrid, tag)),
                 trcDstOffset = spg.getPropertyOffset(dst_tag(NSGrid, tag))] __device__(value_type * shmem, int bid,
                                                                                        int tid) mutable {
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

                    shmem += (tid / tile_size) * arena_size;
                    auto bcoord = spgv._table._activeKeys[blockno];

                    auto block = spgv.block(blockno);
                    for (int cid = tile.thread_rank(); cid < block_size; cid += tile_size) {
                        auto localCoord = spg_t::local_offset_to_coord(cid);
                        auto idx = halo_index(localCoord[0] + 1, localCoord[1] + 1, localCoord[2] + 1);
                        shmem[idx] = block(trcSrcOffset, cid);
                    }

                    // back
                    int bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, -side_length});
                    if (bno >= 0) {
                        auto block = spgv.block(bno);
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 1, j + 1, 0)] =
                                block(trcSrcOffset, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 1}));
                        }
                    } else
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 1, j + 1, 0)] = 0;
                        }
                    // front
                    bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, side_length});
                    if (bno >= 0) {
                        auto block = spgv.block(bno);
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 1, j + 1, halo_side_length - 1)] =
                                block(trcSrcOffset, spg_t::local_coord_to_offset(vec3i{i, j, 0}));
                        }
                    } else
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 1, j + 1, halo_side_length - 1)] = 0;
                        }

                    // bottom
                    bno = spgv._table.tile_query(tile, bcoord + vec3i{0, -side_length, 0});
                    if (bno >= 0) {
                        auto block = spgv.block(bno);
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 1, 0, j + 1)] =
                                block(trcSrcOffset, spg_t::local_coord_to_offset(vec3i{i, side_length - 1, j}));
                        }
                    } else
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 1, 0, j + 1)] = 0;
                        }
                    // up
                    bno = spgv._table.tile_query(tile, bcoord + vec3i{0, side_length, 0});
                    if (bno >= 0) {
                        auto block = spgv.block(bno);
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 1, halo_side_length - 1, j + 1)] =
                                block(trcSrcOffset, spg_t::local_coord_to_offset(vec3i{i, 0, j}));
                        }
                    } else
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 1, halo_side_length - 1, j + 1)] = 0;
                        }

                    // left
                    bno = spgv._table.tile_query(tile, bcoord + vec3i{-side_length, 0, 0});
                    if (bno >= 0) {
                        auto block = spgv.block(bno);
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(0, i + 1, j + 1)] =
                                block(trcSrcOffset, spg_t::local_coord_to_offset(vec3i{side_length - 1, i, j}));
                        }
                    } else
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(0, i + 1, j + 1)] = 0;
                        }
                    // right
                    bno = spgv._table.tile_query(tile, bcoord + vec3i{side_length, 0, 0});
                    if (bno >= 0) {
                        auto block = spgv.block(bno);
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(halo_side_length - 1, i + 1, j + 1)] =
                                block(trcSrcOffset, spg_t::local_coord_to_offset(vec3i{0, i, j}));
                        }
                    } else
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(halo_side_length - 1, i + 1, j + 1)] = 0;
                        }

                    tile.sync();

                    for (int cellno = tile.thread_rank(); cellno < block_size; cellno += tile_size) {
                        auto ccoord = spgv.local_offset_to_coord(cellno);
                        ccoord += 1;

                        float trc_x[2], trc_y[2], trc_z[2];
                        float trc_self = shmem[halo_index(ccoord[0], ccoord[1], ccoord[2])];
                        trc_x[0] = shmem[halo_index(ccoord[0] - 1, ccoord[1], ccoord[2])];
                        trc_x[1] = shmem[halo_index(ccoord[0] + 1, ccoord[1], ccoord[2])];
                        trc_y[0] = shmem[halo_index(ccoord[0], ccoord[1] - 1, ccoord[2])];
                        trc_y[1] = shmem[halo_index(ccoord[0], ccoord[1] + 1, ccoord[2])];
                        trc_z[0] = shmem[halo_index(ccoord[0], ccoord[1], ccoord[2] - 1)];
                        trc_z[1] = shmem[halo_index(ccoord[0], ccoord[1], ccoord[2] + 1)];

                        float trc_xx = scheme::central_diff_2nd(trc_x[0], trc_self, trc_x[1], dx);
                        float trc_yy = scheme::central_diff_2nd(trc_y[0], trc_self, trc_y[1], dx);
                        float trc_zz = scheme::central_diff_2nd(trc_z[0], trc_self, trc_z[1], dx);

                        float diff_term = diffuse * (trc_xx + trc_yy + trc_zz);

                        spgv(trcDstOffset, blockno, cellno) = trc_self + diff_term * dt;
                    }
                });
            pol.shmem(0);

            update_cur(NSGrid, tag);
        }
    }

    void clampDensity(zs::CudaExecutionPolicy &pol, zs::SmallString tag, float clampBelow, ZenoSparseGrid *NSGrid) {

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), rhoOffset = spg.getPropertyOffset(src_tag(NSGrid, tag)),
             clampBelow] __device__(int blockno, int cellno) mutable {
                float rho = spgv.value(rhoOffset, blockno, cellno);
                rho = rho < clampBelow ? 0.f : rho;

                spgv(rhoOffset, blockno, cellno) = rho;
            });
    }

    void coolingTemp(zs::CudaExecutionPolicy &pol, zs::SmallString tag, float coolingRate, float dt,
                     ZenoSparseGrid *NSGrid) {

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), TOffset = spg.getPropertyOffset(src_tag(NSGrid, tag)), coolingRate,
             dt] __device__(int blockno, int cellno) mutable {
                float T = spgv.value(TOffset, blockno, cellno);
                T = zs::max(T - coolingRate * dt, 0.f);

                spgv(TOffset, blockno, cellno) = T;
            });
    }

    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto diffuse = get_input2<float>("Diffusion");
        auto dt = get_input2<float>("dt");
        auto scheme = get_input2<std::string>("Scheme");
        auto wind = get_input2<zeno::vec3f>("WindVelocity");

        auto pol = zs::cuda_exec();
        ///
        if (get_input2<bool>("Density")) {
            compute(pol, "rho", diffuse, dt, scheme, 1.f, wind, NSGrid.get());
        }
        if (get_input2<bool>("Temperature")) {
            compute(pol, "T", diffuse, dt, scheme, 1.f, wind, NSGrid.get());
        }
        if (get_input2<bool>("Fuel")) {
            auto speedScale = get_input2<float>("FuelSpeedScale");
            compute(pol, "fuel", diffuse, dt, scheme, speedScale, wind, NSGrid.get());
        }

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSTracerAdvectDiffuse, {/* inputs: */
                                   {"NSGrid",
                                    "dt",
                                    {gParamType_Bool, "Density", "1"},
                                    {gParamType_Bool, "Temperature", "1"},
                                    {gParamType_Bool, "Fuel", "0"},
                                    {"enum Stencil Semi-Lagrangian MacCormack BFECC", "Scheme", "MacCormack"},
                                    {gParamType_Float, "FuelSpeedScale", "0.05"},
                                    {gParamType_Float, "Diffusion", "0.0"},
                                    {gParamType_Vec3f, "WindVelocity", "0, 0, 0"}},
                                   /* outputs: */
                                   {"NSGrid"},
                                   /* params: */
                                   {},
                                   /* category: */
                                   {"Eulerian"}});

struct ZSTracerEmission : INode {
    void tracerEmit(zs::CudaExecutionPolicy &pol, zs::SmallString tag, ZenoSparseGrid *NSGrid, ZenoSparseGrid *EmitSDF,
                    bool fromObj, float trcAmount, bool addNoise, float noiseAmpPercent, float swirlSize) {

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->spg;
        auto &sdf = EmitSDF->spg;

        auto block_cnt = spg.numBlocks();

        auto dx = spg.voxelSize()[0];

        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), sdfv = zs::proxy<space>(sdf), dx, fromObj, trcAmount, addNoise,
             noiseAmpPercent, swirlSize, tag = src_tag(NSGrid, tag)] __device__(int blockno, int cellno) mutable {
                auto wcoord = spgv.wCoord(blockno, cellno);
                auto emit_sdf = sdfv.wSample("sdf", wcoord);

                if (emit_sdf < 1.1f * dx) {
                    float trcEmitAmount = trcAmount;
                    if (addNoise) {
                        auto pp = (1.f / swirlSize) * wcoord;
                        float fbm = 0, scale = 1;
                        for (int i = 0; i < 3; ++i, pp *= 2.f, scale *= 0.5) {
                            float pln = ZSPerlinNoise1::perlin(pp[0], pp[1], pp[2]);
                            fbm += scale * pln;
                        }
                        trcEmitAmount += noiseAmpPercent * trcAmount * fbm;
                    }
                    spgv(tag, blockno, cellno) = trcEmitAmount;
                }

                if (fromObj) {
                    if (spgv.value("mark", blockno, cellno) > 0.5f) {
                        float trcEmitAmount = trcAmount;
                        if (addNoise) {
                            auto pp = (1.f / swirlSize) * wcoord;
                            float fbm = 0, scale = 1;
                            for (int i = 0; i < 3; ++i, pp *= 2.f, scale *= 0.5) {
                                float pln = ZSPerlinNoise1::perlin(pp[0], pp[1], pp[2]);
                                fbm += scale * pln;
                            }
                            trcEmitAmount += noiseAmpPercent * trcAmount * fbm;
                        }
                        spgv(tag, blockno, cellno) = trcEmitAmount;
                    }
                }
            });
    }

    void velocityEmit(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *NSGrid, ZenoSparseGrid *EmitSDF,
                      ZenoSparseGrid *EmitVel) {

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->spg;
        auto &sdf = EmitSDF->spg;
        auto &vel = EmitVel->spg;

        auto block_cnt = spg.numBlocks();

        auto dx = spg.voxelSize()[0];

        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), sdfv = zs::proxy<space>(sdf), velv = zs::proxy<space>(vel), dx,
             vSrcTag = src_tag(NSGrid, "v")] __device__(int blockno, int cellno) mutable {
                auto wcoord = spgv.wCoord(blockno, cellno);
                auto solid_sdf = sdfv.wSample("sdf", wcoord);

                if (solid_sdf < 1.1f * dx) {
                    auto vel_s = velv.wStaggeredPack("v", wcoord);
                    auto block = spgv.block(blockno);
                    block.template tuple<3>(vSrcTag, cellno) = vel_s;
                }
            });
    }

    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto EmitSDF = get_input<ZenoSparseGrid>("EmitterSDF");
        auto fromObj = get_input2<bool>("fromObjBoundary");
        auto addNoise = get_input2<bool>("AddNoise");
        auto noiseAmpPercent = get_input2<float>("NoiseAmpPercent");
        auto swirlSize = get_input2<float>("SwirlSize");

        if (!NSGrid->getSparseGrid().hasProperty("mark")) {
            fromObj = false;
        }

        auto pol = zs::cuda_exec();

        if (get_input2<bool>("Density")) {
            auto rhoAmount = get_input2<float>("DensityAmount");
            tracerEmit(pol, "rho", NSGrid.get(), EmitSDF.get(), fromObj, rhoAmount, addNoise, noiseAmpPercent,
                       swirlSize);
        }
        if (get_input2<bool>("Temperature")) {
            auto TAmount = get_input2<float>("TemperatureAmount");
            tracerEmit(pol, "T", NSGrid.get(), EmitSDF.get(), fromObj, TAmount, addNoise, noiseAmpPercent, swirlSize);
        }
        if (get_input2<bool>("Fuel")) {
            auto fuelAmount = get_input2<float>("FuelAmount");
            tracerEmit(pol, "fuel", NSGrid.get(), EmitSDF.get(), fromObj, fuelAmount, addNoise, noiseAmpPercent,
                       swirlSize);
        }

        if (get_input2<bool>("hasEmitterVel") && has_input<ZenoSparseGrid>("EmitterVel")) {
            auto EmitVel = get_input<ZenoSparseGrid>("EmitterVel");
            velocityEmit(pol, NSGrid.get(), EmitSDF.get(), EmitVel.get());
        }

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSTracerEmission, {/* inputs: */
                              {"NSGrid",
                               "EmitterSDF",
                               {gParamType_Bool, "hasEmitterVel", "0"},
                               "EmitterVel",
                               {gParamType_Bool, "fromObjBoundary", "0"},
                               {gParamType_Bool, "Density", "1"},
                               {gParamType_Float, "DensityAmount", "1.0"},
                               {gParamType_Bool, "Temperature", "1"},
                               {gParamType_Float, "TemperatureAmount", "1.0"},
                               {gParamType_Bool, "Fuel", "0"},
                               {gParamType_Float, "FuelAmount", "1.0"},
                               {gParamType_Bool, "AddNoise", "0"},
                               {gParamType_Float, "NoiseAmpPercent", "0.2"},
                               {gParamType_Float, "SwirlSize", "0.05"}},
                              /* outputs: */
                              {"NSGrid"},
                              /* params: */
                              {},
                              /* category: */
                              {"Eulerian"}});

struct ZSSmokeBuoyancy : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto dt = get_input2<float>("dt");
        auto gravity = get_input2<zeno::vec3f>("Gravity");
        auto alpha = get_input2<float>("DensityCoef");
        auto beta = get_input2<float>("TemperatureCoef");

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        // add force (accelaration)
        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), dt, alpha, beta, gravity = zs::vec<float, 3>::from_array(gravity),
             vSrcTag = src_tag(NSGrid, "v"), rhoSrcTag = src_tag(NSGrid, "rho"),
             TSrcTag = src_tag(NSGrid, "T")] __device__(int blockno, int cellno) mutable {
                auto icoord = spgv.iCoord(blockno, cellno);

                float rho_this = spgv.value(rhoSrcTag, blockno, cellno);
                float T_this = spgv.value(TSrcTag, blockno, cellno);

                for (int ch = 0; ch < 3; ++ch) {
                    zs::vec<int, 3> offset{0, 0, 0};
                    offset[ch] = -1;

                    auto [bno, cno] = spgv.decomposeCoord(icoord + offset);
                    float rho_face = 0, T_face = 0;
                    if (bno >= 0) {
                        rho_face = spgv.value(rhoSrcTag, bno, cno);
                        T_face = spgv.value(TSrcTag, bno, cno);
                    }

                    rho_face = 0.5f * (rho_this + rho_face);
                    T_face = 0.5f * (T_this + T_face);

                    float G_scale = alpha * rho_face - beta * T_face;

                    spgv(vSrcTag, ch, blockno, cellno) += G_scale * gravity[ch] * dt;
                }
            });

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSSmokeBuoyancy, {/* inputs: */
                             {"NSGrid",
                              "dt",
                              {gParamType_Vec3f, "Gravity", "0, -9.8, 0"},
                              {gParamType_Float, "DensityCoef", "0.0"},
                              {gParamType_Float, "TemperatureCoef", "0.0"}},
                             /* outputs: */
                             {"NSGrid"},
                             /* params: */
                             {},
                             /* category: */
                             {"Eulerian"}});

struct ZSVolumeCombustion : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto dt = get_input2<float>("dt");
        auto ignitionT = get_input2<float>("IgnitionTemperature");
        auto burnSpeed = get_input2<float>("BurnSpeed");
        auto rhoEmitAmount = get_input2<float>("DensityEmitAmount");
        auto TEmitAmount = get_input2<float>("TemperatureEmitAmount");
        auto volumeExpansion = get_input2<float>("VolumeExpansion");

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        // add force (accelaration)
        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), dt, ignitionT, burnSpeed, rhoEmitAmount, TEmitAmount, volumeExpansion,
             fuelOffset = spg.getPropertyOffset(src_tag(NSGrid, "fuel")),
             rhoOffset = spg.getPropertyOffset(src_tag(NSGrid, "rho")),
             TOffset = spg.getPropertyOffset(src_tag(NSGrid, "T")),
             divOffset = spg.getPropertyOffset(src_tag(NSGrid, "div"))] __device__(int blockno, int cellno) mutable {
                float T = spgv.value(TOffset, blockno, cellno);
                float div = 0.f;

                if (T >= ignitionT) {
                    float fuel = spgv.value(fuelOffset, blockno, cellno);
                    float rho = spgv.value(rhoOffset, blockno, cellno);

                    float dF = zs::min(burnSpeed * dt, fuel);
                    fuel -= dF;
                    rho += rhoEmitAmount * dF;
                    T += TEmitAmount * dF;

                    div = volumeExpansion * dF / dt;

                    spgv(fuelOffset, blockno, cellno) = fuel;
                    spgv(rhoOffset, blockno, cellno) = rho;
                    spgv(TOffset, blockno, cellno) = T;
                }

                spgv(divOffset, blockno, cellno) = div;
            });

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSVolumeCombustion, {/* inputs: */
                                {"NSGrid",
                                 "dt",
                                 {gParamType_Float, "IgnitionTemperature", "0.8"},
                                 {gParamType_Float, "BurnSpeed", "0.5"},
                                 {gParamType_Float, "DensityEmitAmount", "0.5"},
                                 {gParamType_Float, "TemperatureEmitAmount", "0.5"},
                                 {gParamType_Float, "VolumeExpansion", "1.2"}},
                                /* outputs: */
                                {"NSGrid"},
                                /* params: */
                                {},
                                /* category: */
                                {"Eulerian"}});

} // namespace zeno