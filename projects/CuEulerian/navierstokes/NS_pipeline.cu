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

namespace zeno {

struct ZSVDBToNavierStokesGrid : INode {
    template <int level>
    using grid_t = typename ZenoSparseGrid::template grid_t<level>;

    void apply() override {
        auto vdbgrid = get_input<VDBFloatGrid>("VDB");

        auto spg = zs::convert_floatgrid_to_sparse_grid(vdbgrid->m_grid, zs::MemoryProperty{zs::memsrc_e::device, -1});
        spg.append_channels(zs::cuda_exec(), {
                                                 {"v0", 3}, // velocity
                                                 {"v1", 3},
                                                 {"p", 1},    // pressure
                                                 {"div", 1},  // velocity divergence
                                                 /*
                                                 {"rho0", 1}, // smoke density
                                                 {"rho1", 1},
                                                 {"T0", 1}, // smoke temperature
                                                 {"T1", 1},
                                                 {"fuel0", 1}, // combustion
                                                 {"fuel1", 1},
                                                 */
                                                 {"tmp", 3}, // FVM, BFECC, MultiGrid, normal
                                                 {"adv", 3}, // reflection
                                                 {"cut", 3}  // cut-cell face weight
                                             });
        spg._background = 0.f;

        auto NSGrid = std::make_shared<ZenoSparseGrid>();
        {
            // initialize multigrid
            auto nbs = spg.numBlocks();
            std::vector<zs::PropertyTag> tags{{"p", 1}, {"tmp", 3}, {"cut", 3}};
            auto transform = spg._transform;

            NSGrid->spg1 = grid_t<1>{spg.get_allocator(), tags, nbs};
            transform.preScale(zs::vec<float, 3>::uniform(2.f));
            NSGrid->spg1._transform = transform;
            NSGrid->spg1._background = spg._background;

            NSGrid->spg2 = grid_t<2>{spg.get_allocator(), tags, nbs};
            transform.preScale(zs::vec<float, 3>::uniform(2.f));
            NSGrid->spg2._transform = transform;
            NSGrid->spg2._background = spg._background;

            NSGrid->spg3 = grid_t<3>{spg.get_allocator(), tags, nbs};
            transform.preScale(zs::vec<float, 3>::uniform(2.f));
            NSGrid->spg3._transform = transform;
            NSGrid->spg3._background = spg._background;

            const auto &table = spg._table;
            auto &table1 = NSGrid->spg1._table;
            auto &table2 = NSGrid->spg2._table;
            auto &table3 = NSGrid->spg3._table;
            table1._cnt.setVal(nbs);
            table2._cnt.setVal(nbs);
            table3._cnt.setVal(nbs);

            auto pol = zs::cuda_exec();
            constexpr auto space = zs::execspace_e::cuda;

            pol(zs::range(nbs),
                [table = zs::proxy<space>(table), tab1 = zs::proxy<space>(table1), tab2 = zs::proxy<space>(table2),
                 tab3 = zs::proxy<space>(table3)] __device__(std::size_t i) mutable {
                    auto bcoord = table._activeKeys[i];
                    tab1.insert(bcoord / 2, i, true);
                    tab2.insert(bcoord / 4, i, true);
                    tab3.insert(bcoord / 8, i, true);
                });
        }
        NSGrid->spg = std::move(spg);
        NSGrid->setMeta("v_cur", 0);
        NSGrid->setMeta("rho_cur", 0);
        NSGrid->setMeta("T_cur", 0);
        NSGrid->setMeta("fuel_cur", 0);

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSVDBToNavierStokesGrid, {/* inputs: */
                                     {"VDB"},
                                     /* outputs: */
                                     {"NSGrid"},
                                     /* params: */
                                     {},
                                     /* category: */
                                     {"Eulerian"}});

struct ZSGridAssignAttribute : INode {
    void apply() override {
        auto ZSGrid = get_input<ZenoSparseGrid>("Grid");
        auto SrcGrid = get_input<ZenoSparseGrid>("SourceGrid");
        auto attrTag = get_input2<std::string>("Attribute");
        auto isStaggered = get_input2<bool>("Staggered");

        auto &spg = ZSGrid->spg;
        auto &src = SrcGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        auto tag = src_tag(ZSGrid, attrTag);
        int nchns = src.getPropertySize(src_tag(SrcGrid, attrTag));
        if (!spg.hasProperty(tag)) {
            spg.append_channels(pol, {{tag, nchns}});
        } else {
            if (nchns != spg.getPropertySize(tag)) {
                throw std::runtime_error(fmt::format("The channel number of [{}] doesn't match!", attrTag));
            }
        }

        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), srcv = zs::proxy<space>(src), nchns, isStaggered,
             srcTag = src_tag(SrcGrid, attrTag), dstTag = tag] __device__(int blockno, int cellno) mutable {
                if (isStaggered) {
                    for (int ch = 0; ch < nchns; ++ch) {
                        auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);
                        spgv(dstTag, ch, blockno, cellno) = srcv.wStaggeredSample(srcTag, ch, wcoord_face);
                    }
                } else {
                    for (int ch = 0; ch < nchns; ++ch) {
                        auto wcoord = spgv.wCoord(blockno, cellno);
                        spgv(dstTag, ch, blockno, cellno) = srcv.wSample(srcTag, ch, wcoord);
                    }
                }
            });

        set_output("Grid", ZSGrid);
    }
};

ZENDEFNODE(ZSGridAssignAttribute, {/* inputs: */
                                   {"Grid", "SourceGrid", {"string", "Attribute", ""}, {"bool", "Staggered", "0"}},
                                   /* outputs: */
                                   {"Grid"},
                                   /* params: */
                                   {},
                                   /* category: */
                                   {"Eulerian"}});

struct ZSNavierStokesDt : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto rho = get_input2<float>("Density");
        auto mu = get_input2<float>("Viscosity");

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();
        auto dx = spg.voxelSize()[0];

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        size_t cell_cnt = block_cnt * spg.block_size;
        zs::Vector<float> res{spg.get_allocator(), count_warps(cell_cnt)};
        zs::memset(zs::mem_device, res.data(), 0, sizeof(float) * count_warps(cell_cnt));

        // maximum velocity
        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), res = zs::proxy<space>(res), cell_cnt,
             vSrcTag = src_tag(NSGrid, "v")] __device__(int blockno, int cellno) mutable {
                float u = spgv.value(vSrcTag, 0, blockno, cellno);
                float v = spgv.value(vSrcTag, 1, blockno, cellno);
                float w = spgv.value(vSrcTag, 2, blockno, cellno);

                size_t cellno_glb = blockno * spgv.block_size + cellno;

                float v_mag = zs::abs(u) + zs::abs(v) + zs::abs(w);

                reduce_max(cellno_glb, cell_cnt, v_mag, res[cellno_glb / 32]);
            });
        float v_max = reduce(pol, res, thrust::maximum<float>{});

        // CFL dt
        float dt_v = dx / (v_max + 1e-10);

        // Viscosity dt
        float nu = mu / (rho + 1e-10); // kinematic viscosity
        int dim = 3;
        float dt_nu = dx * dx / ((2.f * dim * nu) + 1e-10);

        float dt = dt_v < dt_nu ? dt_v : dt_nu;

        fmt::print(fg(fmt::color::blue_violet), "Maximum velocity : {}\n", v_max);
        fmt::print(fg(fmt::color::blue_violet), "CFL time step : {} sec\n", dt);

        set_output("dt", std::make_shared<NumericObject>(dt));
    }
};

ZENDEFNODE(ZSNavierStokesDt, {/* inputs: */
                              {"NSGrid", {"float", "Density", "1.0"}, {"float", "Viscosity", "0.0"}},
                              /* outputs: */
                              {"dt"},
                              /* params: */
                              {},
                              /* category: */
                              {"Eulerian"}});

struct ZSAdvectionScheme : INode {
    void apply() override {
        auto scheme = get_input2<std::string>("Scheme");
        set_output("Scheme", std::make_shared<StringObject>(scheme));
    }
};

ZENDEFNODE(ZSAdvectionScheme, {/* inputs: */
                               {{"enum Stencil Semi-Lagrangian MacCormack BFECC", "Scheme", "MacCormack"}},
                               /* outputs: */
                               {"Scheme"},
                               /* params: */
                               {},
                               /* category: */
                               {"Eulerian"}});

struct ZSNSAdvectDiffuse : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto rho = get_input2<float>("Density");
        auto mu = get_input2<float>("Viscosity");
        auto dt = get_input2<float>("dt");
        auto scheme = get_input2<std::string>("Scheme");
        auto isReflection = get_input2<bool>("Reflection");
        auto wind = get_input2<zeno::vec3f>("WindVelocity");

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();
        auto dx = spg.voxelSize()[0];

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        zs::SmallString advTag;
        if (isReflection) {
            // advection-reflection solver
            advTag = zs::SmallString{"adv"};
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), advTag, vSrcTag = src_tag(NSGrid, "v"),
                 vDstTag = dst_tag(NSGrid, "v")] __device__(int blockno, int cellno) mutable {
                    for (int ch = 0; ch < 3; ++ch) {
                        float u_0 = spgv.value(vSrcTag, ch, blockno, cellno);
                        float u_1 = spgv.value(vDstTag, ch, blockno, cellno);

                        spgv(vSrcTag, ch, blockno, cellno) = 2.f * u_0 - u_1;
                        spgv(advTag, ch, blockno, cellno) = u_0;
                    }
                });
        } else {
            advTag = src_tag(NSGrid, "v");
        }

        if (scheme == "Semi-Lagrangian") {
            // Semi-Lagrangian advection (1st order)
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, wind = zs::vec<float, 3>::from_array(wind), advTag,
                 vSrcTag = src_tag(NSGrid, "v"),
                 vDstTag = dst_tag(NSGrid, "v")] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);

                    for (int ch = 0; ch < 3; ++ch) {
                        auto u_adv = spgv.iStaggeredCellPack(advTag, icoord, ch) + wind;
                        auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);

                        float u_sl = spgv.wStaggeredSample(vSrcTag, ch, wcoord_face - u_adv * dt);

                        spgv(vDstTag, ch, blockno, cellno) = u_sl;
                    }
                });
        } else if (scheme == "MacCormack") {
            // MacCormack scheme
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, wind = zs::vec<float, 3>::from_array(wind), advTag,
                 vSrcTag = src_tag(NSGrid, "v"),
                 vDstTag = zs::SmallString{"tmp"}] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);

                    for (int ch = 0; ch < 3; ++ch) {
                        auto u_adv = spgv.iStaggeredCellPack(advTag, icoord, ch) + wind;
                        auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);

                        float u_sl = spgv.wStaggeredSample(vSrcTag, ch, wcoord_face - u_adv * dt);

                        spgv(vDstTag, ch, blockno, cellno) = u_sl;
                    }
                });
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, wind = zs::vec<float, 3>::from_array(wind), advTag,
                 vTag = src_tag(NSGrid, "v"), vSrcTag = zs::SmallString{"tmp"},
                 vDstTag = dst_tag(NSGrid, "v")] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);

                    for (int ch = 0; ch < 3; ++ch) {
                        auto u_adv = spgv.iStaggeredCellPack(advTag, icoord, ch) + wind;
                        auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);

                        float u_sl = spgv.wStaggeredSample(vSrcTag, ch, wcoord_face + u_adv * dt);

                        float u_mc = spgv.value(vSrcTag, ch, blockno, cellno) +
                                     (spgv.value(vTag, ch, blockno, cellno) - u_sl) / 2.f;

                        // clamp
                        auto icoord_face_src = spgv.worldToIndex(wcoord_face - u_adv * dt);
                        auto arena = spgv.iArena(icoord_face_src, ch);
                        auto sl_mi = arena.minimum(vTag, ch);
                        auto sl_ma = arena.maximum(vTag, ch);
                        if (u_mc > sl_ma || u_mc < sl_mi) {
                            u_mc = arena.isample(vTag, ch, spgv._background);
                        }

                        spgv(vDstTag, ch, blockno, cellno) = u_mc;
                    }
                });
        } else if (scheme == "BFECC") {
            // Back and Forth Error Compensation and Correction (BFECC)
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, wind = zs::vec<float, 3>::from_array(wind), advTag,
                 vSrcTag = src_tag(NSGrid, "v"),
                 vDstTag = dst_tag(NSGrid, "v")] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);

                    for (int ch = 0; ch < 3; ++ch) {
                        auto u_adv = spgv.iStaggeredCellPack(advTag, icoord, ch) + wind;
                        auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);

                        float u_sl = spgv.wStaggeredSample(vSrcTag, ch, wcoord_face - u_adv * dt);

                        spgv(vDstTag, ch, blockno, cellno) = u_sl;
                    }
                });
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, wind = zs::vec<float, 3>::from_array(wind), advTag,
                 vTag = src_tag(NSGrid, "v"), vSrcTag = dst_tag(NSGrid, "v"),
                 vDstTag = zs::SmallString{"tmp"}] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);

                    for (int ch = 0; ch < 3; ++ch) {
                        auto u_adv = spgv.iStaggeredCellPack(advTag, icoord, ch) + wind;
                        auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);

                        float u_sl = spgv.wStaggeredSample(vSrcTag, ch, wcoord_face + u_adv * dt);
                        float u_n = spgv.value(vTag, ch, blockno, cellno);

                        spgv(vDstTag, ch, blockno, cellno) = u_n + (u_n - u_sl) / 2.f;
                    }
                });
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, wind = zs::vec<float, 3>::from_array(wind), advTag,
                 vTag = src_tag(NSGrid, "v"), vSrcTag = zs::SmallString{"tmp"},
                 vDstTag = dst_tag(NSGrid, "v")] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);

                    for (int ch = 0; ch < 3; ++ch) {
                        auto u_adv = spgv.iStaggeredCellPack(advTag, icoord, ch) + wind;
                        auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);
                        auto icoord_face_src = spgv.worldToIndex(wcoord_face - u_adv * dt);
                        auto arena = spgv.iArena(icoord_face_src, ch);

                        float u_sl = arena.isample(vSrcTag, ch, spgv._background);

                        // clamp
                        auto sl_mi = arena.minimum(vTag, ch);
                        auto sl_ma = arena.maximum(vTag, ch);
                        if (u_sl > sl_ma || u_sl < sl_mi) {
                            u_sl = arena.isample(vTag, ch, spgv._background);
                        }

                        spgv(vDstTag, ch, blockno, cellno) = u_sl;
                    }
                });
        } else if (scheme == "Stencil") {
            // shared memory
            using value_type = typename RM_CVREF_T(spg)::value_type;
            constexpr int side_length = RM_CVREF_T(spg)::side_length;
            constexpr int arena_size = (side_length + 4) * (side_length + 4) * (side_length + 4);
            constexpr std::size_t bucket_size = RM_CVREF_T(spg._table)::bucket_size;
            constexpr std::size_t tpb = 4;
            constexpr std::size_t cuda_block_size = bucket_size * tpb;
            pol.shmem(arena_size * sizeof(value_type) * tpb);

            // advection
            pol(zs::Collapse{(block_cnt + tpb - 1) / tpb, cuda_block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, wind = zs::vec<float, 3>::from_array(wind),
                 ts_c = zs::wrapv<bucket_size>{}, tpb_c = zs::wrapv<tpb>{}, blockCnt = block_cnt, advTag,
                 vSrcOffset = spg.getPropertyOffset(src_tag(NSGrid, "v")),
                 vDstOffset = spg.getPropertyOffset(dst_tag(NSGrid, "v"))] __device__(value_type * shmem, int bid,
                                                                                      int tid) mutable {
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

                    for (int ch = 0; ch < 3; ++ch) {
                        // load halo
                        for (int cid = tile.thread_rank(); cid < block_size; cid += tile_size) {
                            auto localCoord = spg_t::local_offset_to_coord(cid);
                            auto idx = halo_index(localCoord[0] + 2, localCoord[1] + 2, localCoord[2] + 2);
                            shmem[idx] = block(vSrcOffset + ch, cid);
                        }

                        // back
                        int bno = spgv._table.tile_query(tile, bcoord + vec3i{0, 0, -side_length});
                        if (bno >= 0) {
                            auto block = spgv.block(bno);
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 2, j + 2, 0)] =
                                    block(vSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 2}));
                                shmem[halo_index(i + 2, j + 2, 1)] =
                                    block(vSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{i, j, side_length - 1}));
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
                                shmem[halo_index(i + 2, j + 2, halo_side_length - 2)] =
                                    block(vSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{i, j, 0}));
                                shmem[halo_index(i + 2, j + 2, halo_side_length - 1)] =
                                    block(vSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{i, j, 1}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 2, j + 2, halo_side_length - 2)] = 0;
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
                                    block(vSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{i, side_length - 2, j}));
                                shmem[halo_index(i + 2, 1, j + 2)] =
                                    block(vSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{i, side_length - 1, j}));
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
                                shmem[halo_index(i + 2, halo_side_length - 2, j + 2)] =
                                    block(vSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{i, 0, j}));
                                shmem[halo_index(i + 2, halo_side_length - 1, j + 2)] =
                                    block(vSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{i, 1, j}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(i + 2, halo_side_length - 2, j + 2)] = 0;
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
                                    block(vSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{side_length - 2, i, j}));
                                shmem[halo_index(1, i + 2, j + 2)] =
                                    block(vSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{side_length - 1, i, j}));
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
                                shmem[halo_index(halo_side_length - 2, i + 2, j + 2)] =
                                    block(vSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{0, i, j}));
                                shmem[halo_index(halo_side_length - 1, i + 2, j + 2)] =
                                    block(vSrcOffset + ch, spg_t::local_coord_to_offset(vec3i{1, i, j}));
                            }
                        } else
                            for (int id = tile.thread_rank(); id < side_area; id += tile_size) {
                                int i = id / side_length;
                                int j = id % side_length;
                                shmem[halo_index(halo_side_length - 2, i + 2, j + 2)] = 0;
                                shmem[halo_index(halo_side_length - 1, i + 2, j + 2)] = 0;
                            }

                        tile.sync();

                        for (int cellno = tile.thread_rank(); cellno < block_size; cellno += tile_size) {
                            auto ccoord = spgv.local_offset_to_coord(cellno);
                            ccoord += 2;

                            auto icoord = spgv.iCoord(blockno, cellno);
                            auto u_adv = spgv.iStaggeredCellPack(advTag, icoord, ch) + wind;

                            const int stcl = 2; // stencil point in each side
                            float u_x[2 * stcl + 1], u_y[2 * stcl + 1], u_z[2 * stcl + 1];

                            for (int i = -stcl; i <= stcl; ++i) {
                                u_x[i + stcl] = shmem[halo_index(ccoord[0] + i, ccoord[1], ccoord[2])];
                                u_y[i + stcl] = shmem[halo_index(ccoord[0], ccoord[1] + i, ccoord[2])];
                                u_z[i + stcl] = shmem[halo_index(ccoord[0], ccoord[1], ccoord[2] + i)];
                            }

                            float adv_term = 0.f;
                            int upwind = u_adv[0] < 0 ? 1 : -1;
                            adv_term += u_adv[0] * scheme::HJ_WENO3(u_x[2 - upwind], u_x[2], u_x[2 + upwind],
                                                                    u_x[2 + 2 * upwind], u_adv[0], dx);
                            upwind = u_adv[1] < 0 ? 1 : -1;
                            adv_term += u_adv[1] * scheme::HJ_WENO3(u_y[2 - upwind], u_y[2], u_y[2 + upwind],
                                                                    u_y[2 + 2 * upwind], u_adv[1], dx);
                            upwind = u_adv[2] < 0 ? 1 : -1;
                            adv_term += u_adv[2] * scheme::HJ_WENO3(u_z[2 - upwind], u_z[2], u_z[2 + upwind],
                                                                    u_z[2 + 2 * upwind], u_adv[2], dx);

                            spgv(vDstOffset + ch, blockno, cellno) = u_adv[ch] - adv_term * dt;
                        }
                    }
                });
            pol.shmem(0);
        } else {
            throw std::runtime_error(fmt::format("Advection scheme [{}] not found!", scheme));
        }

        update_cur(NSGrid, "v");

        if (mu > 0) {
            // diffusion
            pol(zs::range(block_cnt * spg.block_size),
                [spgv = zs::proxy<space>(spg), dx, dt, rho, mu, vSrcTag = src_tag(NSGrid, "v"),
                 vDstTag = dst_tag(NSGrid, "v")] __device__(int cellno) mutable {
                    auto icoord = spgv.iCoord(cellno);

                    for (int ch = 0; ch < 3; ++ch) {
                        const int stcl = 1; // stencil point in each side
                        float u_x[2 * stcl + 1], u_y[2 * stcl + 1], u_z[2 * stcl + 1];

                        for (int i = -stcl; i <= stcl; ++i) {
                            u_x[i + stcl] = spgv.value(vSrcTag, ch, icoord + zs::vec<int, 3>(i, 0, 0));
                            u_y[i + stcl] = spgv.value(vSrcTag, ch, icoord + zs::vec<int, 3>(0, i, 0));
                            u_z[i + stcl] = spgv.value(vSrcTag, ch, icoord + zs::vec<int, 3>(0, 0, i));
                        }

                        float u_xx = scheme::central_diff_2nd(u_x[0], u_x[1], u_x[2], dx);
                        float u_yy = scheme::central_diff_2nd(u_y[0], u_y[1], u_y[2], dx);
                        float u_zz = scheme::central_diff_2nd(u_z[0], u_z[1], u_z[2], dx);

                        float diff_term = mu / rho * (u_xx + u_yy + u_zz);

                        spgv(vDstTag, ch, icoord) = u_x[1] + diff_term * dt;
                    }
                });
            update_cur(NSGrid, "v");
        }

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSNSAdvectDiffuse, {/* inputs: */
                               {"NSGrid",
                                "dt",
                                {"float", "Density", "1.0"},
                                {"float", "Viscosity", "0.0"},
                                {"enum Stencil Semi-Lagrangian MacCormack BFECC", "Scheme", "MacCormack"},
                                {"bool", "Reflection", "0"},
                                {"vec3f", "WindVelocity", "0, 0, 0"}},
                               /* outputs: */
                               {"NSGrid"},
                               /* params: */
                               {},
                               /* category: */
                               {"Eulerian"}});

struct ZSNSExternalForce : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto dt = get_input2<float>("dt");
        auto forceTag = get_input2<std::string>("ForceAttribute");
        auto gravity = get_input2<zeno::vec3f>("Gravity");

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        // add force (accelaration)
        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), gravity = zs::vec<float, 3>::from_array(gravity), dt,
             vSrcTag = src_tag(NSGrid, "v"),
             forceTag = zs::SmallString{forceTag}] __device__(int blockno, int cellno) mutable {
                for (int ch = 0; ch < 3; ++ch) {
                    float acc = (spgv.value(forceTag, ch, blockno, cellno) + gravity[ch]) * dt;
                    spgv(vSrcTag, ch, blockno, cellno) += acc;
                }
            });

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSNSExternalForce, {/* inputs: */
                               {"NSGrid", "dt", {"string", "ForceAttribute", ""}, {"vec3f", "Gravity", "0, 0, 0"}},
                               /* outputs: */
                               {"NSGrid"},
                               /* params: */
                               {},
                               /* category: */
                               {"Eulerian"}});

} // namespace zeno