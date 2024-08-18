#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/LevelSetUtils.tpp"
#include "zensim/geometry/SparseGrid.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

#include "../scheme.hpp"

namespace zeno {
struct ZSRenormalizeSDF : INode {
    void apply() override {
        auto sdfGrid = get_input<ZenoSparseGrid>("Grid");
        auto sdfTag = get_input2<std::string>("SDFAttrName");
        int nIter = get_input2<int>("iterations");

        auto &spg = sdfGrid->spg;
        auto block_cnt = spg.numBlocks();

        if (!spg.hasProperty(sdfTag))
            throw std::runtime_error(fmt::format("the SDFAttribute [{}] does not exist!", sdfTag));

        auto dx = spg.voxelSize()[0];
        auto dt = 0.5 * dx;

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        // shared memory
        using value_type = typename RM_CVREF_T(spg)::value_type;
        constexpr int side_length = RM_CVREF_T(spg)::side_length;
        constexpr int arena_size = (side_length + 4) * (side_length + 4) * (side_length + 4);
        constexpr std::size_t bucket_size = RM_CVREF_T(spg._table)::bucket_size;
        constexpr std::size_t tpb = 4;
        constexpr std::size_t cuda_block_size = bucket_size * tpb;
        pol.shmem(arena_size * sizeof(value_type) * tpb);

        spg.append_channels(pol, {{"tmp", 3}});
        zs::SmallString tagSrc, tagDst;

        for (int iter = 0; iter < nIter; ++iter) {
            if (iter % 2 == 0) {
                tagSrc = sdfTag;
                tagDst = "tmp";
            } else {
                tagSrc = "tmp";
                tagDst = sdfTag;
            }

            pol(zs::Collapse{(block_cnt + tpb - 1) / tpb, cuda_block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, tagSrcOffset = spg.getPropertyOffset(tagSrc),
                 tagDstOffset = spg.getPropertyOffset(tagDst), ts_c = zs::wrapv<bucket_size>{},
                 tpb_c = zs::wrapv<tpb>{},
                 blockCnt = block_cnt] __device__(value_type * shmem, int bid, int tid) mutable {
                    // load halo
                    using vec3i = zs::vec<int, 3>;
                    using spg_t = RM_CVREF_T(spgv);
                    constexpr int side_length = spg_t::side_length;
                    constexpr int side_area = side_length * side_length;
                    constexpr int halo_side_length = side_length + 4;
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
                        shmem[idx] = block(tagSrcOffset, cid);
                    }

                    // back
                    int bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, -side_length});
                    if (bno >= 0) {
                        auto block = spgv.block(bno);
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 2, j + 2, 0)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 2}));
                            shmem[halo_index(i + 2, j + 2, 1)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 1}));
                        }
                    } else
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 2, j + 2, 0)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, j, 1})); // Neumann condition
                            shmem[halo_index(i + 2, j + 2, 1)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, j, 0})); // Neumann condition
                        }
                    // front
                    bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, side_length});
                    if (bno >= 0) {
                        auto block = spgv.block(bno);
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 2, j + 2, halo_side_length - 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, j, 0}));
                            shmem[halo_index(i + 2, j + 2, halo_side_length - 1)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, j, 1}));
                        }
                    } else
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 2, j + 2, halo_side_length - 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 1}));
                            shmem[halo_index(i + 2, j + 2, halo_side_length - 1)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 2}));
                        }

                    // bottom
                    bno = spgv._table.tile_query(tile, bcoord + vec3i{0, -side_length, 0});
                    if (bno >= 0) {
                        auto block = spgv.block(bno);
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 2, 0, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, side_length - 2, j}));
                            shmem[halo_index(i + 2, 1, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, side_length - 1, j}));
                        }
                    } else
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 2, 0, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, 1, j}));
                            shmem[halo_index(i + 2, 1, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, 0, j}));
                        }
                    // up
                    bno = spgv._table.tile_query(tile, bcoord + vec3i{0, side_length, 0});
                    if (bno >= 0) {
                        auto block = spgv.block(bno);
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 2, halo_side_length - 2, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, 0, j}));
                            shmem[halo_index(i + 2, halo_side_length - 1, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, 1, j}));
                        }
                    } else
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(i + 2, halo_side_length - 2, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, side_length - 1, j}));
                            shmem[halo_index(i + 2, halo_side_length - 1, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{i, side_length - 2, j}));
                        }

                    // left
                    bno = spgv._table.tile_query(tile, bcoord + vec3i{-side_length, 0, 0});
                    if (bno >= 0) {
                        auto block = spgv.block(bno);
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(0, i + 2, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{side_length - 2, i, j}));
                            shmem[halo_index(1, i + 2, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{side_length - 1, i, j}));
                        }
                    } else
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(0, i + 2, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{1, i, j}));
                            shmem[halo_index(1, i + 2, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{0, i, j}));
                        }
                    // right
                    bno = spgv._table.tile_query(tile, bcoord + vec3i{side_length, 0, 0});
                    if (bno >= 0) {
                        auto block = spgv.block(bno);
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(halo_side_length - 2, i + 2, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{0, i, j}));
                            shmem[halo_index(halo_side_length - 1, i + 2, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{1, i, j}));
                        }
                    } else
                        for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                            int i = id / side_length;
                            int j = id % side_length;
                            shmem[halo_index(halo_side_length - 2, i + 2, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{side_length - 1, i, j}));
                            shmem[halo_index(halo_side_length - 1, i + 2, j + 2)] =
                                block(tagSrcOffset, spg_t::local_coord_to_offset(vec3i{side_length - 2, i, j}));
                        }

                    tile.sync();

                    for (int cellno = tile.thread_rank(); cellno < block_size; cellno += tile_size) {
                        auto ccoord = spgv.local_offset_to_coord(cellno);
                        ccoord += 2;

                        const int stcl = 2; // stencil point in each side
                        float ls_x[2 * stcl + 1], ls_y[2 * stcl + 1], ls_z[2 * stcl + 1];

                        for (int i = -stcl; i <= stcl; ++i) {
                            ls_x[i + stcl] = shmem[halo_index(ccoord[0] + i, ccoord[1], ccoord[2])];
                            ls_y[i + stcl] = shmem[halo_index(ccoord[0], ccoord[1] + i, ccoord[2])];
                            ls_z[i + stcl] = shmem[halo_index(ccoord[0], ccoord[1], ccoord[2] + i)];
                        }
                        float ls_this = ls_x[stcl];

                        float lx_w = scheme::HJ_WENO3(ls_x[3], ls_x[2], ls_x[1], ls_x[0], 1.0f, dx);
                        float lx_e = scheme::HJ_WENO3(ls_x[1], ls_x[2], ls_x[3], ls_x[4], -1.0f, dx);

                        float ly_w = scheme::HJ_WENO3(ls_y[3], ls_y[2], ls_y[1], ls_y[0], 1.0f, dx);
                        float ly_e = scheme::HJ_WENO3(ls_y[1], ls_y[2], ls_y[3], ls_y[4], -1.0f, dx);

                        float lz_w = scheme::HJ_WENO3(ls_z[3], ls_z[2], ls_z[1], ls_z[0], 1.0f, dx);
                        float lz_e = scheme::HJ_WENO3(ls_z[1], ls_z[2], ls_z[3], ls_z[4], -1.0f, dx);

                        // preventing oscillation
                        float lx_sq, ly_sq, lz_sq;
                        if (ls_this > 0) {
                            // x-direction
                            float lx_m = zs::max(lx_w, 0.f);
                            float lx_p = zs::min(lx_e, 0.f);
                            lx_sq = zs::max(lx_m * lx_m, lx_p * lx_p);
                            // y-direction
                            float ly_m = zs::max(ly_w, 0.f);
                            float ly_p = zs::min(ly_e, 0.f);
                            ly_sq = zs::max(ly_m * ly_m, ly_p * ly_p);
                            // z-direction
                            float lz_m = zs::max(lz_w, 0.f);
                            float lz_p = zs::min(lz_e, 0.f);
                            lz_sq = zs::max(lz_m * lz_m, lz_p * lz_p);
                        } else {
                            // x-direction
                            float lx_m = zs::min(lx_w, 0.f);
                            float lx_p = zs::max(lx_e, 0.f);
                            lx_sq = zs::max(lx_m * lx_m, lx_p * lx_p);
                            // y-direction
                            float ly_m = zs::min(ly_w, 0.f);
                            float ly_p = zs::max(ly_e, 0.f);
                            ly_sq = zs::max(ly_m * ly_m, ly_p * ly_p);
                            // z-direction
                            float lz_m = zs::min(lz_w, 0.f);
                            float lz_p = zs::max(lz_e, 0.f);
                            lz_sq = zs::max(lz_m * lz_m, lz_p * lz_p);
                        }

                        float S0 = ls_x[2] / zs::sqrt(ls_x[2] * ls_x[2] + (lx_sq + ly_sq + lz_sq) * dx * dx);
                        float df = -S0 * (zs::sqrt(lx_sq + ly_sq + lz_sq) - 1.0);

                        spgv(tagDstOffset, blockno, cellno) = ls_this + df * dt;
                    }
                });
        }
        pol.shmem(0);

        if (tagSrc == "tmp") {
            pol(zs::Collapse{spg.numBlocks(), spg.block_size},
                [spgv = zs::proxy<space>(spg), tagSrcOffset = spg.getPropertyOffset("tmp"),
                 tagDstOffset = spg.getPropertyOffset(sdfTag)] __device__(int blockno, int cellno) mutable {
                    spgv(tagDstOffset, blockno, cellno) = spgv.value(tagSrcOffset, blockno, cellno);
                });
        }

        set_output("Grid", sdfGrid);
    }
};

ZENDEFNODE(ZSRenormalizeSDF, {/* inputs: */
                              {"Grid", {gParamType_String, "SDFAttrName", "sdf"}, {gParamType_Int, "iterations", "10"}},
                              /* outputs: */
                              {"Grid"},
                              /* params: */
                              {},
                              /* category: */
                              {"Eulerian"}});

} // namespace zeno