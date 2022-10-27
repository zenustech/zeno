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
struct ZSMakeNavierStokesWorld : INode {
    void apply() override {
        auto zsSPG = zeno::IObject::make<ZenoSparseGrid>();
        auto &spg = zsSPG->spg;

        spg = ZenoSparseGrid::spg_t{
            {{"velocity", 3}, {"vel_n", 3}, {"pressure", 1}, {"pres_n", 1}}, 0, zs::memsrc_e::device, 0};

        set_output("NSgrid", zsSPG);
    }
};

ZENDEFNODE(ZSMakeNavierStokesWorld, {/* inputs: */
                                     {},
                                     /* outputs: */
                                     {"NSgrid"},
                                     /* params: */
                                     {},
                                     /* category: */
                                     {"Eulerian"}});

struct ZSNSAdvectDiffuse : INode {
    void apply() override {
        auto NSgrid = get_input<ZenoSparseGrid>("NSgrid");
        auto rho = get_input2<float>("Density");
        auto mu = get_input2<float>("Viscosity");
        auto dt = get_input2<float>("dt");

        auto &spg = NSgrid->spg;
        auto block_cnt = spg.numBlocks();
        auto dx = spg.voxelSize()[0];

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        // advection
        pol(zs::range(block_cnt * spg.block_size),
            [spgv = zs::proxy<space>(spg), dx, dt] __device__(int cellno) mutable {
                auto icoord = spgv.iCoord(cellno);

                float u_x[5], u_y[5], u_z[5];
                for (int i = -2; i <= 2; ++i) {
                    u_x[i + 2] = spgv.value("velocity", 0, icoord + zs::vec<int, 3>(i, 0, 0));
                    u_y[i + 2] = spgv.value("velocity", 0, icoord + zs::vec<int, 3>(0, i, 0));
                    u_z[i + 2] = spgv.value("velocity", 0, icoord + zs::vec<int, 3>(0, 0, i));
                }
                float u_adv = spgv.value("velocity", 0, icoord);
                float v_adv = spgv.value("velocity", 1, icoord);
                float w_adv = spgv.value("velocity", 2, icoord);

                float adv_term = 0.f;
                int upwind = u_adv < 0 ? 1 : -1;
                adv_term +=
                    u_adv * scheme::HJ_WENO3(u_x[2 - upwind], u_x[2], u_x[2 + upwind], u_x[2 + 2 * upwind], u_adv, dx);
                upwind = v_adv < 0 ? 1 : -1;
                adv_term +=
                    v_adv * scheme::HJ_WENO3(u_y[2 - upwind], u_y[2], u_y[2 + upwind], u_y[2 + 2 * upwind], v_adv, dx);
                upwind = w_adv < 0 ? 1 : -1;
                adv_term +=
                    w_adv * scheme::HJ_WENO3(u_z[2 - upwind], u_z[2], u_z[2 + upwind], u_z[2 + 2 * upwind], w_adv, dx);

                spgv("vel_n", 0, icoord) = u_adv - adv_term * dt;
            });

        // diffusion
        pol(zs::range(block_cnt * spg.block_size),
            [spgv = zs::proxy<space>(spg), dx, dt, rho, mu] __device__(int cellno) mutable {
                auto icoord = spgv.iCoord(cellno);

                float u_x[3], u_y[3], u_z[3];
                for (int i = -1; i <= 1; ++i) {
                    u_x[i + 2] = spgv.value("velocity", 0, icoord + zs::vec<int, 3>(i, 0, 0));
                    u_y[i + 2] = spgv.value("velocity", 0, icoord + zs::vec<int, 3>(0, i, 0));
                    u_z[i + 2] = spgv.value("velocity", 0, icoord + zs::vec<int, 3>(0, 0, i));
                }

                float u_xx = scheme::central_diff_2nd(u_x[0], u_x[1], u_x[2], dx);
                float u_yy = scheme::central_diff_2nd(u_y[0], u_y[1], u_y[2], dx);
                float u_zz = scheme::central_diff_2nd(u_z[0], u_z[1], u_z[2], dx);

                float diff_term = mu / rho * (u_xx + u_yy + u_zz);

                spgv("vel_n", 0, icoord) = u_x[1] + diff_term * dt;
            });
    }
};

} // namespace zeno