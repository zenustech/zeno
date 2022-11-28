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

#include "../utils.cuh"

namespace zeno {

struct ZSPrimitiveToSparseGrid : INode {
    template <zs::kernel_e knl>
    void replace_local_staggered(zs::CudaExecutionPolicy &pol, ZenoParticles *parObjPtr, ZenoSparseGrid *zsSPG,
                                 zs::SmallString dstTag, const int nchns) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &pars = parObjPtr->getParticles();
        auto &spg = zsSPG->getSparseGrid();

        pol(range(pars.size()),
            [spgv = proxy<space>(spg), pars = proxy<space>({}, pars), tagDstOffset = spg.getPropertyOffset(dstTag),
             wOffset = spg.getPropertyOffset("weight"), nchns] __device__(std::size_t pi) mutable {
                auto pos = pars.pack(dim_c<3>, "x", pi);
                for (int d = 0; d < nchns; ++d) { // 0, 1, 2
                    auto arena = spgv.wArena(pos, d, wrapv<knl>{});
                    for (auto loc : ndrange<3>(RM_CVREF_T(arena)::width)) {
                        auto coord = arena.coord(loc);
                        auto [bno, cno] = spgv.decomposeCoord(coord);
                        if (bno < 0) // skip non-exist voxels
                            continue;

                        spgv(tagDstOffset + d, bno, cno) = 0;
                        spgv(wOffset + d, bno, cno) = 0;
                    }
                }
            });
    }

    template <zs::kernel_e knl>
    void replace_local(zs::CudaExecutionPolicy &pol, ZenoParticles *parObjPtr, ZenoSparseGrid *zsSPG,
                       zs::SmallString dstTag, const int nchns) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &pars = parObjPtr->getParticles();
        auto &spg = zsSPG->getSparseGrid();

        pol(range(pars.size()),
            [spgv = proxy<space>(spg), pars = proxy<space>({}, pars), tagDstOffset = spg.getPropertyOffset(dstTag),
             wOffset = spg.getPropertyOffset("weight"), nchns] __device__(std::size_t pi) mutable {
                auto pos = pars.pack(dim_c<3>, "x", pi);
                auto arena = spgv.wArena(pos, wrapv<knl>{});
                for (auto loc : arena.range()) {
                    auto coord = arena.coord(loc);
                    auto [bno, cno] = spgv.decomposeCoord(coord);
                    if (bno < 0) // skip non-exist voxels
                        continue;
#pragma unroll
                    for (int d = 0; d < nchns; ++d) {
                        spgv(tagDstOffset + d, bno, cno) = 0;
                    }
                    spgv(wOffset, bno, cno) = 0;
                }
            });
    }

    template <zs::kernel_e knl>
    void p2g_staggered(zs::CudaExecutionPolicy &pol, ZenoParticles *parObjPtr, ZenoSparseGrid *zsSPG,
                       zs::SmallString parTag, zs::SmallString dstTag, const int nchns) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &pars = parObjPtr->getParticles();
        auto &spg = zsSPG->getSparseGrid();

        pol(range(pars.size()),
            [spgv = proxy<space>(spg), pars = proxy<space>({}, pars), tagSrcOffset = pars.getPropertyOffset(parTag),
             tagDstOffset = spg.getPropertyOffset(dstTag), wOffset = spg.getPropertyOffset("weight"),
             nchns] __device__(std::size_t pi) mutable {
                auto pos = pars.pack(dim_c<3>, "x", pi);
                for (int d = 0; d < nchns; ++d) { // 0, 1, 2
                    auto arena = spgv.wArena(pos, d, wrapv<knl>{});
                    for (auto loc : ndrange<3>(RM_CVREF_T(arena)::width)) {
                        auto coord = arena.coord(loc);
                        auto [bno, cno] = spgv.decomposeCoord(coord);
                        if (bno < 0) // skip non-exist voxels
                            continue;
                        auto W = arena.weight(loc);
                        atomic_add(exec_cuda, &spgv(tagDstOffset + d, bno, cno), W * pars(tagSrcOffset + d, pi));

                        atomic_add(exec_cuda, &spgv(wOffset + d, bno, cno), W);

                        spgv("mark", bno, cno) = 1;
                    }
                }
            });
    }

    template <zs::kernel_e knl>
    void p2g(zs::CudaExecutionPolicy &pol, ZenoParticles *parObjPtr, ZenoSparseGrid *zsSPG, zs::SmallString parTag,
             zs::SmallString dstTag, const int nchns) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &pars = parObjPtr->getParticles();
        auto &spg = zsSPG->getSparseGrid();

        pol(range(pars.size()),
            [spgv = proxy<space>(spg), pars = proxy<space>({}, pars), tagSrcOffset = pars.getPropertyOffset(parTag),
             tagDstOffset = spg.getPropertyOffset(dstTag), wOffset = spg.getPropertyOffset("weight"),
             nchns] __device__(std::size_t pi) mutable {
                auto pos = pars.pack(dim_c<3>, "x", pi);
                auto arena = spgv.wArena(pos, wrapv<knl>{});
                for (auto loc : arena.range()) {
                    auto coord = arena.coord(loc);
                    auto [bno, cno] = spgv.decomposeCoord(coord);
                    if (bno < 0) // skip non-exist voxels
                        continue;
                    auto W = arena.weight(loc);
#pragma unroll
                    for (int d = 0; d < nchns; ++d)
                        atomic_add(exec_cuda, &spgv(tagDstOffset + d, bno, cno), W * pars(tagSrcOffset + d, pi));

                    atomic_add(exec_cuda, &spgv("weight", bno, cno), W);

                    spgv("mark", bno, cno) = 1;
                }
            });
    }

    void apply() override {
        auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
        auto zsSPG = get_input<ZenoSparseGrid>("SparseGrid");
        auto &spg = zsSPG->getSparseGrid();
        auto parTag = get_input2<std::string>("ParticleAttribute");
        auto attrTag = get_input2<std::string>("GridAttribute");
        auto opType = get_input2<std::string>("OpType");
        auto kernel = get_input2<std::string>("Kernel");
        bool isStaggered = get_input2<bool>("staggered");
        bool needInit = get_input2<bool>("initialize");
        bool needNormalize = get_input2<bool>("normalize");

        auto tag = src_tag(zsSPG, attrTag);

        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec().device(0);

        using kt_t = std::variant<wrapv<kernel_e::linear>, wrapv<kernel_e::quadratic>, wrapv<kernel_e::cubic>>;
        kt_t knl;
        if (kernel == "linear")
            knl = wrapv<kernel_e::linear>{};
        else if (kernel == "quadratic")
            knl = wrapv<kernel_e::quadratic>{};
        else if (kernel == "cubic")
            knl = wrapv<kernel_e::cubic>{};
        else
            throw std::runtime_error(fmt::format("Kernel function [{}] not found!", kernel));

        const int nchns = parObjPtrs[0]->getParticles().getPropertySize(parTag);
        for (auto &&parObjPtr : parObjPtrs) {
            int m_nchns = parObjPtr->getParticles().getPropertySize(parTag);
            if (m_nchns != nchns)
                throw std::runtime_error("the size of the ParticleAttribute doesn't match between ZSParticles!");
            if (isStaggered && m_nchns != 3)
                throw std::runtime_error("the size of the ParticleAttribute is not 3!");
        }

        if (!spg.hasProperty(tag)) {
            spg.append_channels(cudaPol, {{tag, nchns}});
        } else {
            int m_nchns = spg.getPropertySize(tag);
            if (isStaggered && m_nchns != 3)
                throw std::runtime_error("the size of the GridAttribute is not 3!");
        }
        ///

        std::vector<PropertyTag> add_tags{{"weight", isStaggered ? 3 : 1}, {"mark", 1}};
        spg.append_channels(cudaPol, add_tags);

        // Initialize (in case of the 1st prim): clear weight, mark
        if (needInit) {
            cudaPol(range(spg.numBlocks() * spg.block_size),
                    [spg = proxy<space>(spg), w_chs = spg.getPropertySize("weight"),
                     wOffset = spg.getPropertyOffset("weight"),
                     markOffset = spg.getPropertyOffset("mark")] __device__(std::size_t cellno) mutable {
                        for (int d = 0; d < w_chs; ++d) {
                            spg(wOffset + d, cellno) = 0;
                        }
                        spg(markOffset, cellno) = 0;
                    });
        }

        if (opType == "replace-all") {
            cudaPol(range(spg.numBlocks() * spg.block_size),
                    [spg = proxy<space>(spg), nchns,
                     tagDstOffset = spg.getPropertyOffset(tag)] __device__(std::size_t cellno) mutable {
                        for (int d = 0; d < nchns; ++d)
                            spg(tagDstOffset + d, cellno) = 0;
                    });
        } else if (opType == "replace-local") {
            for (auto &&parObjPtr : parObjPtrs) {
                if (isStaggered) {
                    match([&](auto knlTag) {
                        constexpr auto knt = decltype(knlTag)::value;
                        replace_local_staggered<knt>(cudaPol, parObjPtr, zsSPG.get(), tag, nchns);
                    })(knl);
                } else {
                    match([&](auto knlTag) {
                        constexpr auto knt = decltype(knlTag)::value;
                        replace_local<knt>(cudaPol, parObjPtr, zsSPG.get(), tag, nchns);
                    })(knl);
                }
            }
        }

        ///
        for (auto &&parObjPtr : parObjPtrs) {
            if (isStaggered) {
                match([&](auto knlTag) {
                    constexpr auto knt = decltype(knlTag)::value;
                    p2g_staggered<knt>(cudaPol, parObjPtr, zsSPG.get(), parTag, tag, nchns);
                })(knl);
            } else {
                match([&](auto knlTag) {
                    constexpr auto knt = decltype(knlTag)::value;
                    p2g<knt>(cudaPol, parObjPtr, zsSPG.get(), parTag, tag, nchns);
                })(knl);
            }
        }

        if (needNormalize) {
            if (isStaggered)
                cudaPol(range(spg.numBlocks() * spg.block_size),
                        [spg = proxy<space>(spg), tagDstOffset = spg._grid.getPropertyOffset(tag),
                         wOffset = spg._grid.getPropertyOffset("weight"),
                         nchns] __device__(std::size_t cellno) mutable {
                            for (int d = 0; d < nchns; ++d) {
                                auto wd = spg(wOffset + d, cellno);
                                if (wd > limits<float>::epsilon() * 10) {
                                    spg(tagDstOffset + d, cellno) /= wd;
                                }
                            }
                        });
            else
                cudaPol(range(spg.numBlocks() * spg.block_size),
                        [spg = proxy<space>(spg), tagDstOffset = spg._grid.getPropertyOffset(tag),
                         wOffset = spg._grid.getPropertyOffset("weight"),
                         nchns] __device__(std::size_t cellno) mutable {
                            auto w = spg(wOffset, cellno);
                            if (w > limits<float>::epsilon() * 10) {
                                for (int d = 0; d < nchns; ++d)
                                    spg(tagDstOffset + d, cellno) /= w;
                            }
                        });
        }

        set_output("SparseGrid", zsSPG);
    }
};

ZENDEFNODE(ZSPrimitiveToSparseGrid, {
                                        /* inputs: */
                                        {"ZSParticles",
                                         "SparseGrid",
                                         {"string", "ParticleAttribute", ""},
                                         {"string", "GridAttribute"},
                                         {"enum replace-all replace-local accumulate", "OpType", "replace-all"},
                                         {"enum linear quadratic cubic", "Kernel", "quadratic"},
                                         {"bool", "staggered", "0"},
                                         {"bool", "initialize", "1"},
                                         {"bool", "normalize", "1"}},
                                        /* outputs: */
                                        {"SparseGrid"},
                                        /* params: */
                                        {},
                                        /* category: */
                                        {"Eulerian"},
                                    });

} // namespace zeno