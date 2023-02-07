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

            pol(zs::Collapse{spg.numBlocks(), spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, tagSrcOffset = spg.getPropertyOffset(tagSrc),
                 tagDstOffset = spg.getPropertyOffset(tagDst)] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);

                    float ls_this = spgv.value(tagSrcOffset, blockno, cellno);
                    float ls_x[5], ls_y[5], ls_z[5];
                    for (int i = -2; i <= 2; ++i) {
                        // stencil with nuemann boundary condition
                        ls_x[i + 2] = spgv.hasVoxel(icoord + zs::vec<int, 3>(i, 0, 0))
                                          ? spgv.value(tagSrcOffset, icoord + zs::vec<int, 3>(i, 0, 0))
                                          : ls_this;
                        ls_y[i + 2] = spgv.hasVoxel(icoord + zs::vec<int, 3>(0, i, 0))
                                          ? spgv.value(tagSrcOffset, icoord + zs::vec<int, 3>(0, i, 0))
                                          : ls_this;
                        ls_z[i + 2] = spgv.hasVoxel(icoord + zs::vec<int, 3>(0, 0, i))
                                          ? spgv.value(tagSrcOffset, icoord + zs::vec<int, 3>(0, 0, i))
                                          : ls_this;
                    }

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
                });
        }

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
                              {"Grid", {"string", "SDFAttrName", "sdf"}, {"int", "iterations", "10"}},
                              /* outputs: */
                              {"Grid"},
                              /* params: */
                              {},
                              /* category: */
                              {"Eulerian"}});

} // namespace zeno