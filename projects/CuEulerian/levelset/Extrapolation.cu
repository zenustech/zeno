#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

#include <zeno/utils/log.h>

#include "../scheme.hpp"
#include "../utils.cuh"

namespace zeno {

struct ZSGridExtrapolateAttr : INode {
    void apply() override {
        auto zsGrid = get_input<ZenoSparseGrid>("Grid");
        auto attrTag = get_input2<std::string>("Attribute");
        auto isStaggered = get_input2<bool>("Staggered");
        auto sdfTag = get_input2<std::string>("SDFAttrName");
        auto direction = get_input2<std::string>("Direction");
        auto maxIter = get_input2<int>("Iterations");

        auto &spg = zsGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        if (!spg.hasProperty(sdfTag))
            throw std::runtime_error(fmt::format("the SDFAttribute [{}] does not exist!", sdfTag));

        auto tag = src_tag(zsGrid, attrTag);
        int nchns = spg.getPropertySize(tag);
        if (isStaggered && nchns != 3)
            throw std::runtime_error("the size of the Attribute is not 3!");

        auto dx = spg.voxelSize()[0];
        auto dt = 0.5f * dx;

        // calculate normal vector
        // "adv" - normal, "tmp" - double buffer, included in NSGrid
        spg.append_channels(pol, {{"adv", 3}, {"tmp", 3}});

        // shared memory
        using value_type = typename RM_CVREF_T(spg)::value_type;
        constexpr int side_length = RM_CVREF_T(spg)::side_length;
        constexpr int arena_size = (side_length + 2) * (side_length + 2) * (side_length + 2);
        constexpr std::size_t bucket_size = RM_CVREF_T(spg._table)::bucket_size;
        constexpr std::size_t tpb = 4;
        constexpr std::size_t cuda_block_size = bucket_size * tpb;
        pol.shmem(arena_size * sizeof(value_type) * tpb);

        pol(zs::Collapse{(block_cnt + tpb - 1) / tpb, cuda_block_size}, [spgv = zs::proxy<space>(spg),
                                                                         sdfOffset = spg.getPropertyOffset(sdfTag), dx,
                                                                         ts_c = zs::wrapv<bucket_size>{},
                                                                         tpb_c = zs::wrapv<tpb>{},
                                                                         blockCnt =
                                                                             block_cnt] __device__(value_type * shmem,
                                                                                                   int bid,
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
                shmem[idx] = block(sdfOffset, cid);
            }

            // back
            int bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, -side_length});
            if (bno >= 0) {
                auto block = spgv.block(bno);
                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                    int i = id / side_length;
                    int j = id % side_length;
                    shmem[halo_index(i + 1, j + 1, 0)] =
                        block(sdfOffset, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 1}));
                }
            } else
                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                    int i = id / side_length;
                    int j = id % side_length;
                    shmem[halo_index(i + 1, j + 1, 0)] =
                        block(sdfOffset, spg_t::local_coord_to_offset(vec3i{i, j, 0})); // Neumann condition
                }
            // front
            bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, side_length});
            if (bno >= 0) {
                auto block = spgv.block(bno);
                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                    int i = id / side_length;
                    int j = id % side_length;
                    shmem[halo_index(i + 1, j + 1, halo_side_length - 1)] =
                        block(sdfOffset, spg_t::local_coord_to_offset(vec3i{i, j, 0}));
                }
            } else
                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                    int i = id / side_length;
                    int j = id % side_length;
                    shmem[halo_index(i + 1, j + 1, halo_side_length - 1)] =
                        block(sdfOffset, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 1}));
                }

            // bottom
            bno = spgv._table.tile_query(tile, bcoord + vec3i{0, -side_length, 0});
            if (bno >= 0) {
                auto block = spgv.block(bno);
                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                    int i = id / side_length;
                    int j = id % side_length;
                    shmem[halo_index(i + 1, 0, j + 1)] =
                        block(sdfOffset, spg_t::local_coord_to_offset(vec3i{i, side_length - 1, j}));
                }
            } else
                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                    int i = id / side_length;
                    int j = id % side_length;
                    shmem[halo_index(i + 1, 0, j + 1)] = block(sdfOffset, spg_t::local_coord_to_offset(vec3i{i, 0, j}));
                }
            // up
            bno = spgv._table.tile_query(tile, bcoord + vec3i{0, side_length, 0});
            if (bno >= 0) {
                auto block = spgv.block(bno);
                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                    int i = id / side_length;
                    int j = id % side_length;
                    shmem[halo_index(i + 1, halo_side_length - 1, j + 1)] =
                        block(sdfOffset, spg_t::local_coord_to_offset(vec3i{i, 0, j}));
                }
            } else
                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                    int i = id / side_length;
                    int j = id % side_length;
                    shmem[halo_index(i + 1, halo_side_length - 1, j + 1)] =
                        block(sdfOffset, spg_t::local_coord_to_offset(vec3i{i, side_length - 1, j}));
                }

            // left
            bno = spgv._table.tile_query(tile, bcoord + vec3i{-side_length, 0, 0});
            if (bno >= 0) {
                auto block = spgv.block(bno);
                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                    int i = id / side_length;
                    int j = id % side_length;
                    shmem[halo_index(0, i + 1, j + 1)] =
                        block(sdfOffset, spg_t::local_coord_to_offset(vec3i{side_length - 1, i, j}));
                }
            } else
                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                    int i = id / side_length;
                    int j = id % side_length;
                    shmem[halo_index(0, i + 1, j + 1)] = block(sdfOffset, spg_t::local_coord_to_offset(vec3i{0, i, j}));
                }
            // right
            bno = spgv._table.tile_query(tile, bcoord + vec3i{side_length, 0, 0});
            if (bno >= 0) {
                auto block = spgv.block(bno);
                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                    int i = id / side_length;
                    int j = id % side_length;
                    shmem[halo_index(halo_side_length - 1, i + 1, j + 1)] =
                        block(sdfOffset, spg_t::local_coord_to_offset(vec3i{0, i, j}));
                }
            } else
                for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                    int i = id / side_length;
                    int j = id % side_length;
                    shmem[halo_index(halo_side_length - 1, i + 1, j + 1)] =
                        block(sdfOffset, spg_t::local_coord_to_offset(vec3i{side_length - 1, i, j}));
                }

            tile.sync();

            for (int cellno = tile.thread_rank(); cellno < block_size; cellno += tile_size) {
                auto ccoord = spgv.local_offset_to_coord(cellno);
                ccoord += 1;

                float sdf_x[2], sdf_y[2], sdf_z[2];
                for (int i = -1; i <= 1; i += 2) {
                    int arrid = (i + 1) >> 1;
                    sdf_x[arrid] = shmem[halo_index(ccoord[0] + i, ccoord[1], ccoord[2])];
                    sdf_y[arrid] = shmem[halo_index(ccoord[0], ccoord[1] + i, ccoord[2])];
                    sdf_z[arrid] = shmem[halo_index(ccoord[0], ccoord[1], ccoord[2] + i)];
                }

                zs::vec<float, 3> normal;
                normal[0] = (sdf_x[1] - sdf_x[0]) / (2.f * dx);
                normal[1] = (sdf_y[1] - sdf_y[0]) / (2.f * dx);
                normal[2] = (sdf_z[1] - sdf_z[0]) / (2.f * dx);

                normal /= zs::max(normal.length(), zs::detail::deduce_numeric_epsilon<float>() * 10);

                spgv._grid.tuple(zs::dim_c<3>, "adv", blockno * spgv.block_size + cellno) = normal;
            }
        });
        pol.shmem(0);

        zs::SmallString tagSrc{tag}, tagDst{"tmp"};
        pol(zs::Collapse{spg.numBlocks(), spg.block_size},
            [spgv = zs::proxy<space>(spg), nchns, tagSrcOffset = spg.getPropertyOffset(tagSrc),
             tagDstOffset = spg.getPropertyOffset(tagDst)] __device__(int blockno, int cellno) mutable {
                for (int ch = 0; ch < nchns; ++ch) {
                    spgv(tagDstOffset + ch, blockno, cellno) = spgv.value(tagSrcOffset + ch, blockno, cellno);
                }
            });

        pol.shmem(arena_size * sizeof(value_type) * tpb);

        for (int iter = 0; iter < maxIter; ++iter) {
            if (iter % 2 == 0) {
                tagSrc = tag;
                tagDst = "tmp";
            } else {
                tagSrc = "tmp";
                tagDst = tag;
            }

            pol(zs::Collapse{(block_cnt + tpb - 1) / tpb, cuda_block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, extDir = zs::SmallString{direction}, nchns, isStaggered,
                 tagSrcOffset = spg.getPropertyOffset(tagSrc), tagDstOffset = spg.getPropertyOffset(tagDst),
                 sdfOffset = spg.getPropertyOffset(sdfTag), ts_c = zs::wrapv<bucket_size>{}, tpb_c = zs::wrapv<tpb>{},
                 blockCnt = block_cnt] __device__(value_type * shmem, int bid, int tid) mutable {
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
                    int bno_neig;

                    for (int ch = 0; ch < nchns; ++ch) {
                        // load halo
                        for (int cid = tile.thread_rank(); cid < block_size; cid += tile_size) {
                            auto localCoord = spg_t::local_offset_to_coord(cid);
                            auto idx = halo_index(localCoord[0] + 1, localCoord[1] + 1, localCoord[2] + 1);
                            shmem[idx] = block(tagSrcOffset + ch, cid);
                        }

                        // back
                        int bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, -side_length});
                        if (ch == 2)
                            bno_neig = bno;
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 1, j + 1, 0)] = block(
                                    tagSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 1}));
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
                                    block(tagSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{i, j, 0}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 1, j + 1, halo_side_length - 1)] = 0;
                            }

                        // bottom
                        bno = spgv._table.tile_query(tile, bcoord + vec3i{0, -side_length, 0});
                        if (ch == 1)
                            bno_neig = bno;
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 1, 0, j + 1)] = block(
                                    tagSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{i, side_length - 1, j}));
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
                                    block(tagSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{i, 0, j}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 1, halo_side_length - 1, j + 1)] = 0;
                            }

                        // left
                        bno = spgv._table.tile_query(tile, bcoord + vec3i{-side_length, 0, 0});
                        if (ch == 0)
                            bno_neig = bno;
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(0, i + 1, j + 1)] = block(
                                    tagSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{side_length - 1, i, j}));
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
                                    block(tagSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{0, i, j}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(halo_side_length - 1, i + 1, j + 1)] = 0;
                            }

                        tile.sync();

                        for (int cellno = tile.thread_rank(); cellno < block_size; cellno += tile_size) {
                            auto m_coord = spg_t::local_offset_to_coord(cellno);
                            auto ccoord = m_coord + 1;

                            float sdf = spgv.value(sdfOffset, blockno, cellno);
                            if (isStaggered && bno_neig >= 0) {
                                auto n_coord = m_coord;
                                n_coord[ch] = (n_coord[ch] - 1) & (side_length - 1);
                                int n_cellno = spg_t::local_coord_to_offset(n_coord);

                                if (m_coord[ch] == 0) {
                                    sdf = 0.5f * (sdf + spgv.value(sdfOffset, bno_neig, n_cellno));
                                } else {
                                    sdf = 0.5f * (sdf + spgv.value(sdfOffset, blockno, n_cellno));
                                }
                            }

                            if ((extDir == "negative" && sdf < -zs::detail::deduce_numeric_epsilon<float>() * 10) ||
                                (extDir == "positive" && sdf > zs::detail::deduce_numeric_epsilon<float>() * 10) || extDir == "both") {

                                zs::vec<float, 3> normal =
                                    spgv._grid.pack(zs::dim_c<3>, "adv", blockno * spgv.block_size + cellno);
                                if (isStaggered && bno_neig >= 0) {
                                    auto n_coord = m_coord;
                                    n_coord[ch] = (n_coord[ch] - 1) & (side_length - 1);
                                    int n_cellno = spg_t::local_coord_to_offset(n_coord);

                                    if (m_coord[ch] == 0) {
                                        normal +=
                                            spgv._grid.pack(zs::dim_c<3>, "adv", bno_neig * spgv.block_size + n_cellno);
                                    } else {
                                        normal +=
                                            spgv._grid.pack(zs::dim_c<3>, "adv", blockno * spgv.block_size + n_cellno);
                                    }

                                    normal /= zs::max(normal.length(), zs::detail::deduce_numeric_epsilon<float>() * 10);
                                }

                                auto sign = [](float val) { return val > 0 ? 1 : -1; };

                                if (extDir == "both")
                                    normal *= sign(sdf);
                                else if (extDir == "negative")
                                    normal *= -1;

                                float v_self = shmem[halo_index(ccoord[0], ccoord[1], ccoord[2])];
                                float v[3][2];
                                for (int i = -1; i <= 1; i += 2) {
                                    int arrid = (i + 1) >> 1;
                                    v[0][arrid] = shmem[halo_index(ccoord[0] + i, ccoord[1], ccoord[2])];
                                    v[1][arrid] = shmem[halo_index(ccoord[0], ccoord[1] + i, ccoord[2])];
                                    v[2][arrid] = shmem[halo_index(ccoord[0], ccoord[1], ccoord[2] + i)];
                                }

                                float adv_term = 0;
                                for (int dim = 0; dim < 3; ++dim) {
                                    adv_term +=
                                        normal[dim] * scheme::upwind_1st(v[dim][0], v_self, v[dim][1], normal[dim], dx);
                                }

                                spgv(tagDstOffset + ch, blockno, cellno) = v_self - adv_term * dt;
                            }
                        }
                    }
                });
        }
        pol.shmem(0);

        if (tagSrc == "tmp") {
            pol(zs::Collapse{spg.numBlocks(), spg.block_size},
                [spgv = zs::proxy<space>(spg), nchns, tagSrcOffset = spg.getPropertyOffset("tmp"),
                 tagDstOffset = spg.getPropertyOffset(tag)] __device__(int blockno, int cellno) mutable {
                    for (int ch = 0; ch < nchns; ++ch) {
                        spgv(tagDstOffset + ch, blockno, cellno) = spgv.value(tagSrcOffset + ch, blockno, cellno);
                    }
                });
        }

        set_output("Grid", zsGrid);
    }
};

ZENDEFNODE(ZSGridExtrapolateAttr, {/* inputs: */
                                   {"Grid",
                                    {gParamType_String, "Attribute", ""},
                                    {gParamType_Bool, "Staggered", "0"},
                                    {gParamType_String, "SDFAttrName", "sdf"},
                                    {"enum both positive negative", "Direction", "positive"},
                                    {gParamType_Int, "Iterations", "5"}},
                                   /* outputs: */
                                   {"Grid"},
                                   /* params: */
                                   {},
                                   /* category: */
                                   {"Eulerian"}});

} // namespace zeno