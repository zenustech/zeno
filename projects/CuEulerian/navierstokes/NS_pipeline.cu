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

namespace zeno {

struct ZSVDBToNavierStokesGrid : INode {
    void apply() override {
        auto vdbgrid = get_input<VDBFloatGrid>("VDB");

        auto spg = zs::convert_floatgrid_to_sparse_grid(vdbgrid->m_grid, zs::MemoryHandle{zs::memsrc_e::device, 0});
        spg.append_channels(zs::cuda_exec(), {{"v0", 3}, {"v1", 3}, {"p0", 1}, {"p1", 1}});
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

                for (int offset = 0; offset != 3; ++offset) {
                    int x = offset;
                    int y = (offset + 1) % 3;
                    int z = (offset + 2) % 3;

                    float u_x[5], u_y[5], u_z[5];
                    for (int i = -2; i <= 2; ++i) {
                        u_x[i + 2] = spgv.value(vSrcTag, x, icoord + zs::vec<int, 3>(i, 0, 0));
                        u_y[i + 2] = spgv.value(vSrcTag, y, icoord + zs::vec<int, 3>(0, i, 0));
                        u_z[i + 2] = spgv.value(vSrcTag, z, icoord + zs::vec<int, 3>(0, 0, i));
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
                    float u_x[3], u_y[3], u_z[3];
                    for (int i = -1; i <= 1; ++i) {
                        u_x[i + 2] = spgv.value(vSrcTag, ch, icoord + zs::vec<int, 3>(i, 0, 0));
                        u_y[i + 2] = spgv.value(vSrcTag, ch, icoord + zs::vec<int, 3>(0, i, 0));
                        u_z[i + 2] = spgv.value(vSrcTag, ch, icoord + zs::vec<int, 3>(0, 0, i));
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
                               {"NSGrid", {"float", "Density", "1.0"}, {"float", "Viscosity", "0.0"}},
                               /* outputs: */
                               {},
                               /* params: */
                               {},
                               /* category: */
                               {"Eulerian"}});

struct ZSNSPressureEvolve : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto rho = get_input2<float>("Density");
        auto Cs = get_input2<float>("Cs");
        auto dt = get_input2<float>("dt");
        int nIter = get_param<int>("iterations");

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();
        auto dx = spg.voxelSize()[0];

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        int &p_cur = NSGrid->readMeta<int &>("p_cur");
        // pressure evolution
        pol(zs::range(block_cnt * spg.block_size),
            [spgv = zs::proxy<space>(spg), dx, dt, rho, Cs,
             pSrcTag = zs::SmallString{std::string("p") + std::to_string(p_cur)},
             pDstTag = zs::SmallString{std::string("p") + std::to_string(p_cur ^ 1)}] __device__(int cellno) mutable {
                auto icoord = spgv.iCoord(cellno);

                
            });
        p_cur ^= 1;
    }
};

} // namespace zeno