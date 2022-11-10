#include "Structures.hpp"
#include "Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/SparseGrid.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

struct ZSPrimitiveToSparseGrid : INode {
    void apply() override {
        auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
        auto zsSPG = get_input<ZenoSparseGrid>("NSGrid");
        auto &spg = zsSPG->getSparseGrid();
        auto tag = zs::SmallString{get_input2<std::string>("Attribute")};
        bool isStaggered = get_input2<bool>("staggered");
        bool needMark = get_input2<bool>("mark");
        bool needClear = get_input2<bool>("clear");

        bool hasDoubleBuffer = false; // TBD

        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        const int nchns = spg._grid.getPropertySize(tag);
        if (isStaggered && nchns != 3)
            throw std::runtime_error("the size of the target staggered property is not 3!");
        ///
        auto cudaPol = cuda_exec().device(0);
        if (needMark) {
            spg.append_channels(cudaPol, {{"w", isStaggered ? 3 : 1}, {"mark", 1}});
        }
        // clear weight, target property and mark (if required)
        if (needClear) {
            cudaPol(range(spg.numBlocks() * spg.block_size),
                    [spg = proxy<space>(spg), tagDstOffset = spg._grid.getPropertyOffset(tag),
                     wOffset = spg._grid.getPropertyOffset("w"), nchns,
                     needMark] __device__(std::size_t cellno) mutable {
                        for (int d = 0; d < nchns; ++d)
                            spg(tagDstOffset + d, cellno) = 0;

                        spg(wOffset, cellno) = 0;

                        if (needMark)
                            spg("mark", cellno / spg.block_size, cellno & (spg.block_size - 1)) = 0;
                    });
        }
        ///
        for (auto &&parObjPtr : parObjPtrs) {
            auto &pars = parObjPtr->getParticles();
            if (isStaggered) {
                cudaPol(range(pars.size()), [spgv = proxy<space>(spg), pars = proxy<space>({}, pars),
                                             tagSrcOffset = pars.getPropertyOffset(tag),
                                             tagDstOffset = spg._grid.getPropertyOffset(tag),
                                             wOffset = spg._grid.getPropertyOffset("w"), nchns,
                                             needMark] __device__(std::size_t pi) mutable {
                    using spg_t = RM_CVREF_T(spgv);
                    auto pos = pars.pack(dim_c<3>, "x", pi);
                    for (int d = 0; d < nchns; ++d) { // 0, 1, 2
                        auto arena = spgv.wArena(pos, d, wrapv<kernel_e::quadratic>{});
                        for (auto loc : ndrange<3>(RM_CVREF_T(arena)::width)) {
                            auto coord = arena.coord(loc);
                            auto [bno, cno] = spgv.decomposeCoord(coord);
                            if (bno < 0) // skip non-exist voxels
                                continue;
                            auto W = arena.weight(loc);
                            atomic_add(exec_cuda, &spgv(tagDstOffset + d, bno, cno), W * pars(tagSrcOffset + d, pi));

                            atomic_add(exec_cuda, &spgv(wOffset + d, bno, cno), W);

                            if (needMark)
                                spgv("mark", bno, cno) = 1;
                        }
                    }
                });
            } else {
                cudaPol(range(pars.size()), [spgv = proxy<space>(spg), pars = proxy<space>({}, pars),
                                             tagSrcOffset = pars.getPropertyOffset(tag),
                                             tagDstOffset = spg._grid.getPropertyOffset(tag),
                                             wOffset = spg._grid.getPropertyOffset("w"), nchns,
                                             needMark] __device__(std::size_t pi) mutable {
                    using spg_t = RM_CVREF_T(spgv);
                    auto pos = pars.pack(dim_c<3>, "x", pi);
                    auto arena = spgv.wArena(pos, wrapv<kernel_e::quadratic>{});
                    for (auto loc : arena.range()) {
                        auto coord = arena.coord(loc);
                        auto [bno, cno] = spgv.decomposeCoord(coord);
                        if (bno < 0) // skip non-exist voxels
                            continue;
                        auto W = arena.weight(loc);
#pragma unroll
                        for (int d = 0; d < nchns; ++d)
                            atomic_add(exec_cuda, &spgv(tagDstOffset + d, bno, cno), W * pars(tagSrcOffset + d, pi));

                        atomic_add(exec_cuda, &spgv("w", bno, cno), W);

                        if (needMark)
                            spgv("mark", bno, cno) = 1;
                    }
                });
            }
        }
        if (isStaggered)
            cudaPol(range(spg.numBlocks() * spg.block_size),
                    [spg = proxy<space>(spg), tagDstOffset = spg._grid.getPropertyOffset(tag),
                     wOffset = spg._grid.getPropertyOffset("w"), nchns] __device__(std::size_t cellno) mutable {
                        for (int d = 0; d < nchns; ++d) {
                            auto wd = spg(wOffset + d, cellno);
                            if (wd > limits<float>::epsilon() * 10)
                                spg(tagDstOffset + d, cellno) /= wd;
                            else
                                spg(tagDstOffset + d, cellno) = 0;
                        }
                    });
        else
            cudaPol(range(spg.numBlocks() * spg.block_size),
                    [spg = proxy<space>(spg), tagDstOffset = spg._grid.getPropertyOffset(tag),
                     wOffset = spg._grid.getPropertyOffset("w"), nchns] __device__(std::size_t cellno) mutable {
                        auto w = spg(wOffset, cellno);
                        if (w > limits<float>::epsilon() * 10) {
                            for (int d = 0; d < nchns; ++d)
                                spg(tagDstOffset + d, cellno) /= w;
                        } else {
                            for (int d = 0; d < nchns; ++d)
                                spg(tagDstOffset + d, cellno) = 0;
                        }
                    });
    }
};

ZENDEFNODE(ZSPrimitiveToSparseGrid, {
                                        {"ZSParticles",
                                         "NSGrid",
                                         {"string", "Attribute", "v"},
                                         {"bool", "staggered", "0"},
                                         {"bool", "clear", "1"},
                                         {"bool", "mark", "0"}},
                                        {"NSGrid"},
                                        {},
                                        {"Hybrid"},
                                    });

} // namespace zeno