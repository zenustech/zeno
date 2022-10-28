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
        int nIter = get_param<int>("iterations");
        auto sdfGrid = get_input<ZenoSparseGrid>("SDF");

        auto &sdf = sdfGrid->spg;
        auto block_cnt = sdf.numBlocks();

        ZenoSparseGrid::spg_t sdf_tmp{{{"sdf", 1}}, 1, zs::memsrc_e::device, 0};
        // topo copy
        sdf_tmp._table = sdf._table;
        sdf_tmp._transform = sdf._transform;
        sdf_tmp._background = sdf._background;
        sdf_tmp._grid.resize(sdf.numBlocks() * sdf.block_size);

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        for (int iter = 0; iter < nIter; ++iter) {
            pol(zs::range(block_cnt * sdf.block_size),
                [sdfv = zs::proxy<space>(sdf), sdf_tmpv = zs::proxy<space>(sdf_tmp)] __device__(int cellno) mutable {
                    auto icoord = sdfv.iCoord(cellno);
                    auto dx = sdfv.voxelSize()[0];
                    auto dt = 0.5 * dx;

                    float ls_x[5], ls_y[5], ls_z[5];
                    for (int i = -2; i <= 2; ++i) {
                        ls_x[i + 2] = sdfv.value("sdf", icoord + zs::vec<int, 3>(i, 0, 0));
                        ls_y[i + 2] = sdfv.value("sdf", icoord + zs::vec<int, 3>(0, i, 0));
                        ls_z[i + 2] = sdfv.value("sdf", icoord + zs::vec<int, 3>(0, 0, i));
                    }
                    float ls_this = sdfv.value("sdf", icoord);

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

                    sdf_tmpv("sdf", icoord) = ls_this + df * dt;
                });
            std::swap(sdf, sdf_tmp);
        }

        set_output("SDF", sdfGrid);
    }
};

ZENDEFNODE(ZSRenormalizeSDF, {/* inputs: */
                              {"SDF"},
                              /* outputs: */
                              {"SDF"},
                              /* params: */
                              {{"int", "iterations", "10"}},
                              /* category: */
                              {"Eulerian"}});

} // namespace zeno