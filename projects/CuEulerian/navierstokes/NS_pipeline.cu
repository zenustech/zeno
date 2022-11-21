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
    template <int level> using grid_t = typename ZenoSparseGrid::template grid_t<level>;

    void apply() override {
        auto vdbgrid = get_input<VDBFloatGrid>("VDB");

        auto spg = zs::convert_floatgrid_to_sparse_grid(vdbgrid->m_grid, zs::MemoryHandle{zs::memsrc_e::device, 0});
        spg.append_channels(zs::cuda_exec(), {
                                                 {"v0", 3},
                                                 {"v1", 3},
                                                 {"p0", 1},
                                                 {"p1", 1},
                                                 {"rho0", 1}, // smoke density
                                                 {"rho1", 1},
                                                 {"T0", 1}, // smoke temperature
                                                 {"T1", 1},
                                                 {"tmp", 3}, // FVM, BFECC, MultiGrid
                                                 {"adv", 3}  // reflection
                                             });
        spg._background = 0.f;

        auto NSGrid = std::make_shared<ZenoSparseGrid>();
        {
            // initialize multigrid
            auto nbs = spg.numBlocks();
            std::vector<zs::PropertyTag> tags{{"p0", 1}, {"p1", 1}, {"tmp", 3}};
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
        }
        NSGrid->spg = std::move(spg);
        NSGrid->setMeta("v_cur", 0);
        NSGrid->setMeta("p_cur", 0);
        NSGrid->setMeta("rho_cur", 0);
        NSGrid->setMeta("T_cur", 0);

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

struct ZSNSAssignAttribute : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto SrcGrid = get_input<ZenoSparseGrid>("SourceGrid");
        auto tag = get_input2<std::string>("Attribute");
        auto isStaggered = get_input2<bool>("Staggered");

        auto &spg = NSGrid->spg;
        auto &src = SrcGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        auto num_ch = src.getPropertySize(tag);
        if (num_ch != spg.getPropertySize(src_tag(NSGrid, tag))) {
            throw std::runtime_error(fmt::format("The channel number of [{}] doesn't match!", tag));
        }

        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), srcv = zs::proxy<space>(src), num_ch, isStaggered,
             srcTag = src_tag(SrcGrid, tag),
             dstTag = src_tag(NSGrid, tag)] __device__(int blockno, int cellno) mutable {
                if (isStaggered) {
                    for (int ch = 0; ch < num_ch; ++ch) {
                        auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);
                        spgv(dstTag, ch, blockno, cellno) = srcv.wStaggeredSample(srcTag, ch, wcoord_face);
                    }
                } else {
                    for (int ch = 0; ch < num_ch; ++ch) {
                        auto wcoord = spgv.wCoord(blockno, cellno);
                        spgv(dstTag, ch, blockno, cellno) = srcv.wSample(srcTag, ch, wcoord);
                    }
                }
            });

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSNSAssignAttribute, {/* inputs: */
                                 {"NSGrid", "SourceGrid", {"string", "Attribute", ""}, {"bool", "Staggered", "0"}},
                                 /* outputs: */
                                 {"NSGrid"},
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

struct ZSNSAdvectDiffuse : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto rho = get_input2<float>("Density");
        auto mu = get_input2<float>("Viscosity");
        auto dt = get_input2<float>("dt");
        auto scheme = get_input2<std::string>("Scheme");
        auto isReflection = get_input2<bool>("Reflection");

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
                [spgv = zs::proxy<space>(spg), dx, dt, advTag, vSrcTag = src_tag(NSGrid, "v"),
                 vDstTag = dst_tag(NSGrid, "v")] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    auto wcoord = spgv.indexToWorld(icoord);

                    for (int ch = 0; ch < 3; ++ch) {
                        zs::vec<float, 3> u_adv;
                        u_adv[0] = spgv.iStaggeredCellSample(advTag, 0, icoord, ch);
                        u_adv[1] = spgv.iStaggeredCellSample(advTag, 1, icoord, ch);
                        u_adv[2] = spgv.iStaggeredCellSample(advTag, 2, icoord, ch);

                        auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);

                        float u_sl = spgv.wStaggeredSample(vSrcTag, ch, wcoord_face - u_adv * dt);

                        spgv(vDstTag, ch, blockno, cellno) = u_sl;
                    }
                });
        } else if (scheme == "BFECC") {
            // Back and Forth Error Compensation and Correction (BFECC)
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, advTag, vSrcTag = src_tag(NSGrid, "v"),
                 vDstTag = dst_tag(NSGrid, "v")] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    auto wcoord = spgv.indexToWorld(icoord);

                    for (int ch = 0; ch < 3; ++ch) {
                        zs::vec<float, 3> u_adv;
                        u_adv[0] = spgv.iStaggeredCellSample(advTag, 0, icoord, ch);
                        u_adv[1] = spgv.iStaggeredCellSample(advTag, 1, icoord, ch);
                        u_adv[2] = spgv.iStaggeredCellSample(advTag, 2, icoord, ch);

                        auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);

                        float u_sl = spgv.wStaggeredSample(vSrcTag, ch, wcoord_face - u_adv * dt);

                        spgv(vDstTag, ch, blockno, cellno) = u_sl;
                    }
                });
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, advTag, vTag = src_tag(NSGrid, "v"),
                 vSrcTag = dst_tag(NSGrid, "v"),
                 vDstTag = zs::SmallString{"tmp"}] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    auto wcoord = spgv.indexToWorld(icoord);

                    for (int ch = 0; ch < 3; ++ch) {
                        zs::vec<float, 3> u_adv;
                        u_adv[0] = spgv.iStaggeredCellSample(advTag, 0, icoord, ch);
                        u_adv[1] = spgv.iStaggeredCellSample(advTag, 1, icoord, ch);
                        u_adv[2] = spgv.iStaggeredCellSample(advTag, 2, icoord, ch);

                        auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);

                        float u_sl = spgv.wStaggeredSample(vSrcTag, ch, wcoord_face + u_adv * dt);
                        float u_n = spgv.value(vTag, ch, blockno, cellno);

                        spgv(vDstTag, ch, blockno, cellno) = u_n + (u_n - u_sl) / 2.f;
                    }
                });
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, advTag, vTag = src_tag(NSGrid, "v"),
                 vSrcTag = zs::SmallString{"tmp"},
                 vDstTag = dst_tag(NSGrid, "v")] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    auto wcoord = spgv.indexToWorld(icoord);

                    for (int ch = 0; ch < 3; ++ch) {
                        zs::vec<float, 3> u_adv;
                        u_adv[0] = spgv.iStaggeredCellSample(advTag, 0, icoord, ch);
                        u_adv[1] = spgv.iStaggeredCellSample(advTag, 1, icoord, ch);
                        u_adv[2] = spgv.iStaggeredCellSample(advTag, 2, icoord, ch);

                        auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);
                        auto wcoord_face_src = wcoord_face - u_adv * dt;

                        float u_sl = spgv.wStaggeredSample(vSrcTag, ch, wcoord_face_src);

                        // clamp
                        auto arena = spgv.wArena(wcoord_face_src, ch);
                        auto sl_mi = arena.minimum(vTag, ch);
                        auto sl_ma = arena.maximum(vTag, ch);
                        if (u_sl > sl_ma || u_sl < sl_mi) {
                            u_sl = spgv.wStaggeredSample(vTag, ch, wcoord_face_src);
                        }

                        spgv(vDstTag, ch, blockno, cellno) = u_sl;
                    }
                });
        } else if (scheme == "Finite-Difference") {
            // advection
            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, advTag, vSrcTag = src_tag(NSGrid, "v"),
                 vDstTag = dst_tag(NSGrid, "v")] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);

                    for (int ch = 0; ch < 3; ++ch) {
                        zs::vec<float, 3> u_adv;
                        u_adv[0] = spgv.iStaggeredCellSample(advTag, 0, icoord, ch);
                        u_adv[1] = spgv.iStaggeredCellSample(advTag, 1, icoord, ch);
                        u_adv[2] = spgv.iStaggeredCellSample(advTag, 2, icoord, ch);

                        int x = ch;
                        int y = (ch + 1) % 3;
                        int z = (ch + 2) % 3;

                        const int stcl = 2; // stencil point in each side
                        float u_x[2 * stcl + 1], u_y[2 * stcl + 1], u_z[2 * stcl + 1];

                        zs::vec<int, 3> offset;

                        for (int i = -stcl; i <= stcl; ++i) {
                            offset = zs::vec<int, 3>::zeros();
                            offset[x] = i;
                            u_x[i + stcl] = spgv.value(vSrcTag, x, icoord + offset);

                            offset = zs::vec<int, 3>::zeros();
                            offset[y] = i;
                            u_y[i + stcl] = spgv.value(vSrcTag, x, icoord + offset);

                            offset = zs::vec<int, 3>::zeros();
                            offset[z] = i;
                            u_z[i + stcl] = spgv.value(vSrcTag, x, icoord + offset);
                        }

                        float adv_term = 0.f;
                        int upwind = u_adv[x] < 0 ? 1 : -1;
                        adv_term += u_adv[x] * scheme::HJ_WENO3(u_x[2 - upwind], u_x[2], u_x[2 + upwind],
                                                                u_x[2 + 2 * upwind], u_adv[x], dx);
                        upwind = u_adv[y] < 0 ? 1 : -1;
                        adv_term += u_adv[y] * scheme::HJ_WENO3(u_y[2 - upwind], u_y[2], u_y[2 + upwind],
                                                                u_y[2 + 2 * upwind], u_adv[y], dx);
                        upwind = u_adv[z] < 0 ? 1 : -1;
                        adv_term += u_adv[z] * scheme::HJ_WENO3(u_z[2 - upwind], u_z[2], u_z[2 + upwind],
                                                                u_z[2 + 2 * upwind], u_adv[z], dx);

                        spgv(vDstTag, ch, blockno, cellno) = u_adv[x] - adv_term * dt;
                    }
                });
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
                                {"enum Finite-Difference Semi-Lagrangian BFECC", "Scheme", "Finite-Difference"},
                                {"bool", "Reflection", "0"}},
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
            [spgv = zs::proxy<space>(spg), gravity, dt, vSrcTag = src_tag(NSGrid, "v"),
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

struct ZSNSNaiveSolidWall : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto SolidSDF = get_input<ZenoSparseGrid>("SolidSDF");
        auto SolidVel = get_input<ZenoSparseGrid>("SolidVel");

        auto &sdf = SolidSDF->spg;
        auto &vel = SolidVel->spg;
        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), sdfv = zs::proxy<space>(sdf), velv = zs::proxy<space>(vel),
             vSrcTag = src_tag(NSGrid, "v")] __device__(int blockno, int cellno) mutable {
                auto wcoord = spgv.wCoord(blockno, cellno);
                auto solid_sdf = sdfv.wSample("sdf", wcoord);

                if (solid_sdf < 0) {
                    auto vel_s = velv.wStaggeredPack("v", wcoord);
                    auto block = spgv.block(blockno);
                    block.template tuple<3>(vSrcTag, cellno) = vel_s;
                }

                spgv("sdf", blockno, cellno) = solid_sdf;
            });

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSNSNaiveSolidWall, {/* inputs: */
                                {"NSGrid", "SolidSDF", "SolidVel"},
                                /* outputs: */
                                {"NSGrid"},
                                /* params: */
                                {},
                                /* category: */
                                {"Eulerian"}});

} // namespace zeno