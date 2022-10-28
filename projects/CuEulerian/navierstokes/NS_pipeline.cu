#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/LevelSetUtils.tpp"
#include "zensim/geometry/SparseGrid.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
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
        spg.append_channels(zs::cuda_exec(), {{"v0", 3}, {"v1", 3}, {"p0", 1}, {"p1", 1}, {"div_v", 1}});
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
        float dt_v = CFL * dx / v_max;

        // Viscosity dt
        float nu = mu / rho; // kinematic viscosity
        int dim = 3;
        float dt_nu = CFL * 0.5f * dim * dx * dx / nu;

        float dt = dt_v < dt_nu ? dt_v : dt_nu;

        fmt::print(fg(fmt::color::blue_violet), "time step : {} sec\n", dt);

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
                        u_y[i + stcl] = spgv.value(vSrcTag, y, icoord + zs::vec<int, 3>(0, i, 0));
                        u_z[i + stcl] = spgv.value(vSrcTag, z, icoord + zs::vec<int, 3>(0, 0, i));
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
        int nIter = get_param<int>("iterations");

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

        int &p_cur = NSGrid->readMeta<int &>("p_cur");

        float dxSqrOverDt = dx * dx / dt;

        // pressure Poisson equation - point Jacobi iteration
        for (int iter = 0; iter < nIter; ++iter) {
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
        }

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
                                 {"NSGrid", "dt", {"float", "Density", "1.0"}},
                                 /* outputs: */
                                 {},
                                 /* params: */
                                 {{"int", "iterations", "10"}},
                                 /* category: */
                                 {"Eulerian"}});

} // namespace zeno