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
    void apply() override {
        auto vdbgrid = get_input<VDBFloatGrid>("VDB");

        auto spg = zs::convert_floatgrid_to_sparse_grid(vdbgrid->m_grid, zs::MemoryHandle{zs::memsrc_e::device, 0});
        spg.append_channels(zs::cuda_exec(), {{"v0", 3},
                                              {"v1", 3},
                                              {"p0", 1},
                                              {"p1", 1},
                                              {"div_v", 1},
                                              {"rho", 1}, // smoke density
                                              {"T", 1},   // smoke temperature
                                              {"flux", 3}});
        spg._background = 0.f;

        auto zsSPG = std::make_shared<ZenoSparseGrid>();
        zsSPG->spg = std::move(spg);
        zsSPG->setMeta("v_cur", 0);
        zsSPG->setMeta("p_cur", 0);

        set_output("NSGrid", zsSPG);
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

        int &v_cur = NSGrid->readMeta<int &>("v_cur");

        // maximum velocity
        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), res = zs::proxy<space>(res), cell_cnt,
             vSrcTag = zs::SmallString{std::string("v") + std::to_string(v_cur)}] __device__(int blockno,
                                                                                             int cellno) mutable {
                float u = spgv.value(vSrcTag, 0, blockno, cellno);
                float v = spgv.value(vSrcTag, 1, blockno, cellno);
                float w = spgv.value(vSrcTag, 2, blockno, cellno);

                size_t cellno_glb = blockno * spgv.block_size + cellno;

                float v_mag = zs::abs(u) + zs::abs(v) + zs::abs(w);

                reduce_max(cellno_glb, cell_cnt, v_mag, res[cellno_glb / 32]);
            });
        float v_max = reduce(pol, res, thrust::maximum<float>{});

        // CFL dt
        const float CFL = 0.8f;
        float dt_v = CFL * dx / (v_max + 1e-10);

        // Viscosity dt
        float nu = mu / (rho + 1e-10); // kinematic viscosity
        int dim = 3;
        float dt_nu = CFL * dx * dx / ((2.f * dim * nu) + 1e-10);

        float dt = dt_v < dt_nu ? dt_v : dt_nu;

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

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();
        auto dx = spg.voxelSize()[0];

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        int &v_cur = NSGrid->readMeta<int &>("v_cur");

        // advection
        pol(zs::range(block_cnt * spg.block_size),
            [spgv = zs::proxy<space>(spg), dx, dt, vSrcTag = zs::SmallString{std::string("v") + std::to_string(v_cur)},
             vDstTag = zs::SmallString{std::string("v") + std::to_string(v_cur ^ 1)}] __device__(int cellno) mutable {
                auto icoord = spgv.iCoord(cellno);

                for (int ch = 0; ch < 3; ++ch) {
                    int x = ch;
                    int y = (ch + 1) % 3;
                    int z = (ch + 2) % 3;

                    const int stcl = 2; // stencil point in each side
                    float u_x[2 * stcl + 1], u_y[2 * stcl + 1], u_z[2 * stcl + 1];

                    for (int i = -stcl; i <= stcl; ++i) {
                        u_x[i + stcl] = spgv.value(vSrcTag, x, icoord + zs::vec<int, 3>(i, 0, 0));
                        u_y[i + stcl] = spgv.value(vSrcTag, x, icoord + zs::vec<int, 3>(0, i, 0));
                        u_z[i + stcl] = spgv.value(vSrcTag, x, icoord + zs::vec<int, 3>(0, 0, i));
                    }

                    float u_adv = spgv.value(vSrcTag, x, icoord);
                    float v_adv = spgv.value(vSrcTag, y, icoord);
                    float w_adv = spgv.value(vSrcTag, z, icoord);

                    float adv_term = 0.f;
                    int upwind = u_adv < 0 ? 1 : -1;
                    adv_term += u_adv * scheme::HJ_WENO3(u_x[2 - upwind], u_x[2], u_x[2 + upwind], u_x[2 + 2 * upwind],
                                                         u_adv, dx);
                    upwind = v_adv < 0 ? 1 : -1;
                    adv_term += v_adv * scheme::HJ_WENO3(u_y[2 - upwind], u_y[2], u_y[2 + upwind], u_y[2 + 2 * upwind],
                                                         v_adv, dx);
                    upwind = w_adv < 0 ? 1 : -1;
                    adv_term += w_adv * scheme::HJ_WENO3(u_z[2 - upwind], u_z[2], u_z[2 + upwind], u_z[2 + 2 * upwind],
                                                         w_adv, dx);

                    spgv(vDstTag, x, icoord) = u_adv - adv_term * dt;
                }
            });
        v_cur ^= 1;

        // diffusion
        pol(zs::range(block_cnt * spg.block_size),
            [spgv = zs::proxy<space>(spg), dx, dt, rho, mu,
             vSrcTag = zs::SmallString{std::string("v") + std::to_string(v_cur)},
             vDstTag = zs::SmallString{std::string("v") + std::to_string(v_cur ^ 1)}] __device__(int cellno) mutable {
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
        v_cur ^= 1;
    }
};

ZENDEFNODE(ZSNSAdvectDiffuse, {/* inputs: */
                               {"NSGrid", "dt", {"float", "Density", "1.0"}, {"float", "Viscosity", "0.0"}},
                               /* outputs: */
                               {},
                               /* params: */
                               {},
                               /* category: */
                               {"Eulerian"}});

struct ZSNSExternalForce : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto dt = get_input2<float>("dt");
        auto force = get_input2<zeno::vec3f>("Force");

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        int &v_cur = NSGrid->readMeta<int &>("v_cur");

        // add force (accelaration)
        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), force, dt,
             vSrcTag = zs::SmallString{std::string("v") + std::to_string(v_cur)}] __device__(int blockno,
                                                                                             int cellno) mutable {
                for (int ch = 0; ch < 3; ++ch)
                    spgv(vSrcTag, ch, blockno, cellno) += force[ch] * dt;
            });
    }
};

ZENDEFNODE(ZSNSExternalForce, {/* inputs: */
                               {"NSGrid", "dt", {"vec3f", "Force", "0, 0, 0"}},
                               /* outputs: */
                               {},
                               /* params: */
                               {},
                               /* category: */
                               {"Eulerian"}});

struct ZSNSPressureProject : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto rho = get_input2<float>("Density");
        auto dt = get_input2<float>("dt");
        int nIter = get_input2<int>("iterations");

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();
        auto dx = spg.voxelSize()[0];

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        int &v_cur = NSGrid->readMeta<int &>("v_cur");

        // velocity divergence
        pol(zs::range(block_cnt * spg.block_size),
            [spgv = zs::proxy<space>(spg), dx, dt,
             vSrcTag = zs::SmallString{std::string("v") + std::to_string(v_cur)}] __device__(int cellno) mutable {
                auto icoord = spgv.iCoord(cellno);

                float u_x[2], u_y[2], u_z[2];
                for (int i = 0; i <= 1; ++i) {
                    u_x[i] = spgv.value(vSrcTag, 0, icoord + zs::vec<int, 3>(i, 0, 0));
                    u_y[i] = spgv.value(vSrcTag, 1, icoord + zs::vec<int, 3>(0, i, 0));
                    u_z[i] = spgv.value(vSrcTag, 2, icoord + zs::vec<int, 3>(0, 0, i));
                }

                float div_term = ((u_x[1] - u_x[0]) + (u_y[1] - u_y[0]) + (u_z[1] - u_z[0])) / dx;

                spgv("div_v", icoord) = div_term;
            });

        const float dxSqrOverDt = dx * dx / dt;

        int &p_cur = NSGrid->readMeta<int &>("p_cur");

        // zs::CppTimer timer;
        // timer.tick();
        // pressure Poisson equation
        for (int iter = 0; iter < nIter; ++iter) {
#if 0
            // point Jacobi iteration
            pol(zs::range(block_cnt * spg.block_size),
                [spgv = zs::proxy<space>(spg), dxSqrOverDt, rho,
                 pSrcTag = zs::SmallString{std::string("p") + std::to_string(p_cur)},
                 pDstTag =
                     zs::SmallString{std::string("p") + std::to_string(p_cur ^ 1)}] __device__(int cellno) mutable {
                    auto icoord = spgv.iCoord(cellno);

                    float div = spgv.value("div_v", icoord);

                    const int stcl = 1; // stencil point in each side
                    float p_x[2 * stcl + 1], p_y[2 * stcl + 1], p_z[2 * stcl + 1];

                    for (int i = -stcl; i <= stcl; ++i) {
                        p_x[i + stcl] = spgv.value(pSrcTag, icoord + zs::vec<int, 3>(i, 0, 0));
                        p_y[i + stcl] = spgv.value(pSrcTag, icoord + zs::vec<int, 3>(0, i, 0));
                        p_z[i + stcl] = spgv.value(pSrcTag, icoord + zs::vec<int, 3>(0, 0, i));
                    }

                    float p_this =
                        -(div * dxSqrOverDt * rho - (p_x[0] + p_x[2] + p_y[0] + p_y[2] + p_z[0] + p_z[2])) / 6.f;

                    spgv(pDstTag, icoord) = p_this;
                });
            p_cur ^= 1;
#else
            // red-black SOR iteration
            const float sor = 1.2f; // over relaxation rate

            for (int clr = 0; clr != 2; ++clr) {

                pol(zs::range(block_cnt * 32), [spgv = zs::proxy<space>(spg), dxSqrOverDt, rho, clr,
                                                sor] __device__(int tid) mutable {
                    auto blockno = tid / 32;

                    auto bcoord = spgv._table._activeKeys[blockno];
                    if ((((bcoord[0] & 8) ^ (bcoord[1] & 8) ^ (bcoord[2] & 8)) >> 3) == clr)
                        return;

                    auto tile = zs::cg::tiled_partition<32>(zs::cg::this_thread_block());
                    for (int c_clr = 0; c_clr != 2; ++c_clr) {

                        for (int cno = tile.thread_rank(); cno < 256; cno += tile.num_threads()) {
                            auto cellno = (cno << 1) | c_clr;

                            auto ccoord = spgv.local_offset_to_coord(cellno);
                            auto icoord = bcoord + ccoord;

                            float div = spgv.value("div_v", blockno, cellno);

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
                                    ((p_x[0] + p_x[2] + p_y[0] + p_y[2] + p_z[0] + p_z[2]) - div * dxSqrOverDt * rho) /
                                    6.f;

                            spgv("p0", blockno, cellno) = p_this;
                        }
                    }
                });
            }
#endif
        }
        // timer.tock("jacobi/sor iterations");

        // pressure projection
        pol(zs::range(block_cnt * spg.block_size),
            [spgv = zs::proxy<space>(spg), dx, dt, rho,
             vSrcTag = zs::SmallString{std::string("v") + std::to_string(v_cur)},
             vDstTag = zs::SmallString{std::string("v") + std::to_string(v_cur ^ 1)},
             pSrcTag = zs::SmallString{std::string("p") + std::to_string(p_cur)}] __device__(int cellno) mutable {
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
        v_cur ^= 1;
    }
};

ZENDEFNODE(ZSNSPressureProject, {/* inputs: */
                                 {"NSGrid", "dt", {"int", "iterations", "10"}, {"float", "Density", "1.0"}},
                                 /* outputs: */
                                 {},
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

        int &v_cur = NSGrid->readMeta<int &>("v_cur");

        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), sdfv = zs::proxy<space>(sdf), velv = zs::proxy<space>(vel),
             vSrcTag = zs::SmallString{std::string("v") + std::to_string(v_cur)}] __device__(int blockno,
                                                                                             int cellno) mutable {
                auto wcoord = spgv.wCoord(blockno, cellno);
                auto solid_sdf = sdfv.wSample("sdf", wcoord);

                if (solid_sdf < 0) {
                    auto vel_s = velv.wStaggeredPack("v", wcoord);
                    auto block = spgv.block(blockno);
                    block.template tuple<3>(vSrcTag, cellno) = vel_s;
                }

                spgv("sdf", blockno, cellno) = solid_sdf;
            });
    }
};

ZENDEFNODE(ZSNSNaiveSolidWall, {/* inputs: */
                                {"NSGrid", "SolidSDF", "SolidVel"},
                                /* outputs: */
                                {},
                                /* params: */
                                {},
                                /* category: */
                                {"Eulerian"}});

struct ZSTracerAdvectDiffuse : INode {
    void compute(zs::CudaExecutionPolicy &pol, zs::SmallString tag, float diffuse, float dt, ZenoSparseGrid *NSGrid) {

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto dx = spg.voxelSize()[0];

        int &v_cur = NSGrid->readMeta<int &>("v_cur");

        // Finite Volume Method (FVM)
        // numrtical flux of tracer
        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), diffuse, dx, tag,
             vSrcTag = zs::SmallString{std::string("v") + std::to_string(v_cur)}] __device__(int blockno,
                                                                                             int cellno) mutable {
                auto icoord = spgv.iCoord(blockno, cellno);

                const int stcl = 2; // stencil point in each side
                float trc[3][2 * stcl];

                // | i - 2 | i - 1 | i | i + 1 |
                for (int i = -stcl; i < stcl; ++i) {
                    trc[0][i + stcl] = spgv.value(tag, icoord + zs::vec<int, 3>(i, 0, 0));
                    trc[1][i + stcl] = spgv.value(tag, icoord + zs::vec<int, 3>(0, i, 0));
                    trc[2][i + stcl] = spgv.value(tag, icoord + zs::vec<int, 3>(0, 0, i));
                }

                float u_adv[3];
                for (int ch = 0; ch < 3; ++ch)
                    u_adv[ch] = spgv.value(vSrcTag, ch, icoord);

                // approximate value at i - 1/2
                float flux[3];
                for (int ch = 0; ch < 3; ++ch) {
                    // convection flux
                    if (u_adv[ch] < 0)
                        flux[ch] = u_adv[ch] * scheme::TVD_MUSCL3(trc[ch][1], trc[ch][2], trc[ch][3]);
                    else
                        flux[ch] = u_adv[ch] * scheme::TVD_MUSCL3(trc[ch][2], trc[ch][1], trc[ch][0]);

                    // diffusion flux
                    flux[ch] -= diffuse * (trc[ch][2] - trc[ch][1]) / dx;
                }

                for (int ch = 0; ch < 3; ++ch) {
                    spgv("flux", ch, blockno, cellno) = flux[ch];
                }
            });

        // time integration of tracer
        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), dx, dt, tag] __device__(int blockno, int cellno) mutable {
                auto icoord = spgv.iCoord(blockno, cellno);

                float flux[3][2];
                for (int ch = 0; ch < 3; ++ch) {
                    zs::vec<int, 3> offset{0, 0, 0};
                    offset[ch] = 1;

                    flux[ch][0] = spgv.value("flux", ch, icoord);
                    flux[ch][1] = spgv.value("flux", ch, icoord + offset);
                }

                float dtrc = 0;
                for (int ch = 0; ch < 3; ++ch) {
                    dtrc += (flux[ch][0] - flux[ch][1]) / dx;
                }
                dtrc *= dt;

                spgv(tag, blockno, cellno) += dtrc;
            });
    }

    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto diffuse = get_input2<float>("Diffusion");
        auto dt = get_input2<float>("dt");

        auto pol = zs::cuda_exec();
        ///
        if (get_input2<bool>("Density"))
            compute(pol, "rho", diffuse, dt, NSGrid.get());
        if (get_input2<bool>("Temperature"))
            compute(pol, "T", diffuse, dt, NSGrid.get());
    }
};

ZENDEFNODE(ZSTracerAdvectDiffuse,
           {/* inputs: */
            {"NSGrid", "dt", {"float", "Diffusion", "0.0"}, {"bool", "Density", "1"}, {"bool", "Temperature", "1"}},
            /* outputs: */
            {},
            /* params: */
            {},
            /* category: */
            {"Eulerian"}});

struct ZSTracerEmission : INode {
    void compute(zs::CudaExecutionPolicy &pol, zs::SmallString tag, ZenoSparseGrid *NSGrid, ZenoSparseGrid *EmitSDF) {

        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = NSGrid->spg;
        auto &sdf = EmitSDF->spg;

        auto block_cnt = spg.numBlocks();

        auto dx = spg.voxelSize()[0];

        pol(zs::Collapse{block_cnt, spg.block_size}, [spgv = zs::proxy<space>(spg), sdfv = zs::proxy<space>(sdf), dx,
                                                      tag] __device__(int blockno, int cellno) mutable {
            auto wcoord = spgv.wCoord(blockno, cellno);
            auto emit_sdf = sdfv.wSample("sdf", wcoord);

            if (emit_sdf <= 1.5f * dx) {
                // fix me: naive emission
                spgv(tag, blockno, cellno) = 1.0;
            }
        });
    }

    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto EmitSDF = get_input<ZenoSparseGrid>("EmitterSDF");

        auto pol = zs::cuda_exec();

        if (get_input2<bool>("Density"))
            compute(pol, "rho", NSGrid.get(), EmitSDF.get());
        if (get_input2<bool>("Temperature"))
            compute(pol, "T", NSGrid.get(), EmitSDF.get());
    }
};

ZENDEFNODE(ZSTracerEmission, {/* inputs: */
                              {"NSGrid", "EmitterSDF", {"bool", "Density", "1"}, {"bool", "Temperature", "1"}},
                              /* outputs: */
                              {},
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
        auto T_amb = get_input2<float>("AmbientTemp");
        auto T_ref = get_input2<float>("ReferenceTemp");

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        int &v_cur = NSGrid->readMeta<int &>("v_cur");

        // add force (accelaration)
        pol(zs::Collapse{block_cnt, spg.block_size},
            [spgv = zs::proxy<space>(spg), dt, gravity, alpha, beta, T_amb, T_ref,
             vSrcTag = zs::SmallString{std::string("v") + std::to_string(v_cur)}] __device__(int blockno,
                                                                                             int cellno) mutable {
                auto icoord = spgv.iCoord(blockno, cellno);

                float rho_this = spgv.value("rho", blockno, cellno);
                float T_this = spgv.value("T", blockno, cellno);

                for (int ch = 0; ch < 3; ++ch) {
                    zs::vec<int, 3> offset{0, 0, 0};
                    offset[ch] = -1;

                    float rho_face = spgv.value("rho", icoord + icoord);
                    float T_face = spgv.value("T", icoord + icoord);

                    rho_face = 0.5f * (rho_this + rho_face);
                    T_face = 0.5f * (T_this + T_face);

                    float G_scale = alpha * rho_face - beta * (T_face * T_ref - T_amb);

                    spgv(vSrcTag, ch, blockno, cellno) += G_scale * gravity[ch] * dt;
                }
            });
    }
};

ZENDEFNODE(ZSSmokeBuoyancy, {/* inputs: */
                             {"NSGrid",
                              "dt",
                              {"vec3f", "Gravity", "0, -9.8, 0"},
                              {"float", "DensityCoef", "0.0"},
                              {"float", "TemperatureCoef", "0.0"},
                              {"float", "AmbientTemp", "273.0"},
                              {"float", "ReferenceTemp", "3000.0"}},
                             /* outputs: */
                             {},
                             /* params: */
                             {},
                             /* category: */
                             {"Eulerian"}});

struct ZSExtendSparseGrid : INode {

    template <typename PredT> void refit(ZenoSparseGrid *nsgridPtr, std::string tag, PredT pred) {
        using namespace zs;
        static constexpr auto space = execspace_e::cuda;
        namespace cg = ::cooperative_groups;
        auto pol = cuda_exec();
        auto &spg = nsgridPtr->getSparseGrid();
        // make sure spg.block_size % 32 == 0

        auto nbs = spg.numBlocks();
        using Ti = RM_CVREF_T(nbs);

        Vector<Ti> marks{spg.get_allocator(), nbs + 1}, offsets{spg.get_allocator(), nbs + 1};
        marks.reset(0);

        pol(range(nbs * 32), [spg = proxy<space>(spg), tagOffset = spg.getPropertyOffset(tag),
                              marks = proxy<space>(marks), pred] __device__(std::size_t i) mutable {
            auto tile = cg::tiled_partition<32>(cg::this_thread_block());
            auto bno = i / 32;
            auto cellno = tile.thread_rank();

            while (cellno < spg.block_size) {
                if (tile.ballot(pred(spg(tagOffset, bno, cellno))))
                    break;
                cellno += 32;
            }
            if (tile.thread_rank() == 0 && cellno < spg.block_size)
                marks[bno] = 1;
        });

        exclusive_scan(pol, std::begin(marks), std::end(marks), std::begin(offsets));
        auto newNbs = offsets.getVal(nbs);
        fmt::print("compacting {} blocks to {} active blocks.\n", nbs, newNbs);

        /// @brief compact active blocks
        // grid
        auto &grid = spg._grid;
        auto dstgrid = grid;
        // table
        auto &table = spg._table;
        table.reset(false);
        table._cnt.setVal(newNbs);
        auto &keys = table._activeKeys;
        auto newKeys = keys;

        pol(range(nbs * spg._table.bucket_size),
            [grid = proxy<space>(grid), dstgrid = proxy<space>(dstgrid), marks = proxy<space>(marks),
             newKeys = proxy<space>(newKeys), keys = proxy<space>(keys), table = proxy<space>(table),
             offsets = proxy<space>(offsets),
             bs_c = wrapv<RM_CVREF_T(spg)::block_size>{}] __device__(std::size_t i) mutable {
                constexpr auto block_size = decltype(bs_c)::value;
                constexpr auto bucket_size = RM_CVREF_T(table)::bucket_size;
                static_assert(block_size % bucket_size == 0, "block_size should be a multiple of bucket_size");
                auto tile = cg::tiled_partition<bucket_size>(cg::this_thread_block());
                auto bno = i / bucket_size;
                if (marks[bno] == 0)
                    return;
                auto dstBno = offsets[bno];
                // grid
                for (auto cellno = tile.thread_rank(); cellno < block_size; cellno += bucket_size) {
                    for (int chn = 0; chn != grid.numChannels(); ++chn)
                        dstgrid(chn, dstBno, cellno) = grid(chn, bno, cellno);
                }
                // table
                auto bcoord = keys[bno];
                table.tile_insert(tile, bcoord, dstBno, false); // do not enqueue key, hence set false
                if (tile.thread_rank() == 0)
                    newKeys[dstBno] = bcoord;
            });
        grid = std::move(dstgrid);
        keys = std::move(newKeys);
    }

    template <typename PredT>
    void extend(ZenoSparseGrid *nsgridPtr, std::string tag, std::size_t &nbsOffset, PredT pred) {
        using namespace zs;
        static constexpr auto space = execspace_e::cuda;
        namespace cg = ::cooperative_groups;
        auto pol = cuda_exec();
        auto &spg = nsgridPtr->getSparseGrid();
        // make sure spg.block_size % 32 == 0

        auto nbs = spg.numBlocks() - nbsOffset;
        if (nbs == 0)
            return;
        // worst case is that all candidate blocks activate all surrounding neighbors
        spg.resize(pol, nbs * 26 + nbsOffset);

        // zeno::log_info("currently {} blocks (offset {}), resizing to {}\n", nbsOffset + nbs, nbsOffset,
        //                nbs * 26 + nbsOffset);

        if (!spg._grid.hasProperty(tag))
            throw std::runtime_error(fmt::format("property [{}] not exist!", tag));

        pol(range(nbs * spg._table.bucket_size), [spg = proxy<space>(spg), tagOffset = spg.getPropertyOffset(tag),
                                                  nbsOffset, pred] __device__(std::size_t i) mutable {
            auto tile = cg::tiled_partition<RM_CVREF_T(spg._table)::bucket_size>(cg::this_thread_block());
            auto bno = i / spg._table.bucket_size + nbsOffset;
            auto cellno = tile.thread_rank();
            // searching for active voxels within this block

            while (cellno < spg.block_size) {
                if (tile.ballot(pred(spg(tagOffset, bno, cellno))))
                    break;
                cellno += spg._table.bucket_size;
            }
            if (cellno < spg.block_size) {
                auto bcoord = spg.iCoord(bno, 0);
                for (auto loc : ndrange<3>(3)) {
                    auto dir = make_vec<int>(loc) - 1;
                    // spg._table.insert(bcoord + dir * spg.side_length);
                    spg._table.tile_insert(tile, bcoord + dir * spg.side_length, RM_CVREF_T(spg._table)::sentinel_v,
                                           true);
                }
            }
        });
        // [ nbsOffset | nbsOffset + nbs | spg.numBlocks() ]
        nbsOffset += nbs;
        auto newNbs = spg.numBlocks();
        newNbs -= nbsOffset;
        if (newNbs > 0)
            zs::memset(mem_device, (void *)spg._grid.tileOffset(nbsOffset), 0,
                       (std::size_t)newNbs * spg._grid.tileBytes());

        if (tag == "sdf")
            pol(range(newNbs * spg.block_size),
                [dx = spg.voxelSize()[0], spg = proxy<space>(spg), sdfOffset = spg.getPropertyOffset("sdf"),
                 blockOffset = nbsOffset * spg.block_size] __device__(std::size_t cellno) mutable {
                    spg(sdfOffset, blockOffset + cellno) = 3 * dx;
                });
    }

    void apply() override {
        auto zsSPG = get_input<ZenoSparseGrid>("NSGrid");
        auto tag = get_input2<std::string>("Attribute");
        auto nlayers = get_input2<int>("layers");
        auto needRefit = get_input2<bool>("refit");

        std::size_t nbs = 0;
        int opt = 0;
        if (tag == "rho")
            opt = 1;
        else if (tag == "sdf")
            opt = 2;

        if (needRefit && opt != 0) {
            if (opt == 1)
                refit(zsSPG.get(), tag,
                      [] __device__(float v) -> bool { return v > zs::limits<float>::epsilon() * 10; });
            else if (opt == 2)
                refit(zsSPG.get(), tag,
                      [dx = zsSPG->getSparseGrid().voxelSize()[0]] __device__(float v) -> bool { return v < 2 * dx; });
            opt = 0;
        }

        while (nlayers-- > 0) {
            if (opt == 0)
                extend(zsSPG.get(), tag, nbs, [] __device__(float v) { return true; });
            else if (opt == 1)
                extend(zsSPG.get(), tag, nbs,
                       [] __device__(float v) -> bool { return v > zs::limits<float>::epsilon() * 10; });
            else if (opt == 2)
                extend(zsSPG.get(), tag, nbs,
                       [dx = zsSPG->getSparseGrid().voxelSize()[0]] __device__(float v) -> bool { return v < 2 * dx; });
            opt = 0; // always active since
        }

        set_output("NSGrid", zsSPG);
    }
};

ZENDEFNODE(ZSExtendSparseGrid,
           {/* inputs: */
            {"NSGrid", {"enum rho sdf", "Attribute", "rho"}, {"bool", "refit", "1"}, {"int", "layers", "2"}},
            /* outputs: */
            {"NSGrid"},
            /* params: */
            {},
            /* category: */
            {"Eulerian"}});

} // namespace zeno