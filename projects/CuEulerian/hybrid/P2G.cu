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
#include "zeno/utils/log.h"

namespace zeno {

struct ZSPrimitiveToSparseGrid : INode {
    template <zs::kernel_e knl>
    void activate_block(zs::CudaExecutionPolicy &pol, ZenoParticles *parObjPtr, ZenoSparseGrid *zsSPG, size_t num_pars,
                        bool multigrid) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &pars = parObjPtr->getParticles();
        auto &spg = zsSPG->getSparseGrid();

        spg.resizePartition(pol, num_pars + spg.numBlocks());
        spg._table._buildSuccess.setVal(1);
        pol(range(pars.size()),
            [spgv = proxy<space>(spg), pars = proxy<space>({}, pars)] __device__(std::size_t pi) mutable {
                using spg_t = RM_CVREF_T(spgv);

                auto pos = pars.pack(dim_c<3>, "x", pi);
                auto arena = spgv.wArena(pos, wrapv<knl>{});
                for (auto loc : arena.range()) {
                    auto coord = arena.coord(loc);
                    auto bcoord = coord - (coord & (spg_t::side_length - 1));
                    spgv._table.insert(bcoord);
                }
            });

        if (int tag = spg._table._buildSuccess.getVal(); tag == 0)
            zeno::log_error("check P2G activate success state: {}\n", tag);

        const auto nbs = spg.numBlocks();
        spg.resizeGrid(nbs);

        if (multigrid) {
            /// @brief adjust multigrid accordingly
            // grid
            auto &spg1 = zsSPG->spg1;
            spg1.resize(pol, nbs);
            auto &spg2 = zsSPG->spg2;
            spg2.resize(pol, nbs);
            auto &spg3 = zsSPG->spg3;
            spg3.resize(pol, nbs);
            // table
            {
                const auto &table = spg._table;
                auto &table1 = spg1._table;
                auto &table2 = spg2._table;
                auto &table3 = spg3._table;
                table1.reset(true);
                table1._cnt.setVal(nbs);
                table2.reset(true);
                table2._cnt.setVal(nbs);
                table3.reset(true);
                table3._cnt.setVal(nbs);

                table1._buildSuccess.setVal(1);
                table2._buildSuccess.setVal(1);
                table3._buildSuccess.setVal(1);
                pol(range(nbs), [table = proxy<space>(table), tab1 = proxy<space>(table1), tab2 = proxy<space>(table2),
                                 tab3 = proxy<space>(table3)] __device__(std::size_t i) mutable {
                    auto bcoord = table._activeKeys[i];
                    tab1.insert(bcoord / 2, i, true);
                    tab2.insert(bcoord / 4, i, true);
                    tab3.insert(bcoord / 8, i, true);
                });

                int tag1 = table1._buildSuccess.getVal();
                int tag2 = table2._buildSuccess.getVal();
                int tag3 = table3._buildSuccess.getVal();
                if (tag1 == 0 || tag2 == 0 || tag3 == 0)
                    zeno::log_error("check P2G multigrid activate success state: {}, {}, {}\n", tag1, tag2, tag3);
            }
        }
    }

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
        bool needActivate = get_input2<bool>("activateBlock");
        bool multigrid = get_input2<bool>("activateMultiGrid");

        auto tag = src_tag(zsSPG, attrTag);

        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto cudaPol = cuda_exec();

        using kt_t = std::variant<wrapv<kernel_e::linear>, wrapv<kernel_e::quadratic>, wrapv<kernel_e::cubic>,
                                  wrapv<kernel_e::delta2>, wrapv<kernel_e::delta3>, wrapv<kernel_e::delta4>>;
        kt_t knl;
        if (kernel == "linear")
            knl = wrapv<kernel_e::linear>{};
        else if (kernel == "quadratic")
            knl = wrapv<kernel_e::quadratic>{};
        else if (kernel == "cubic")
            knl = wrapv<kernel_e::cubic>{};
        else if (kernel == "delta2")
            knl = wrapv<kernel_e::delta2>{};
        else if (kernel == "delta3")
            knl = wrapv<kernel_e::delta3>{};
        else if (kernel == "delta4")
            knl = wrapv<kernel_e::delta4>{};
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

        // Activate blocks containing particles
        if (needActivate) {
            // count total number of particles
            size_t num_pars = 0;
            for (auto &&parObjPtr : parObjPtrs) {
                num_pars += parObjPtr->numParticles();
            }
            for (auto &&parObjPtr : parObjPtrs) {
                match([&](auto knlTag) {
                    constexpr auto knt = decltype(knlTag)::value;
                    activate_block<knt>(cudaPol, parObjPtr, zsSPG.get(), num_pars, multigrid);
                })(knl);
            }
        }

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
                                if (wd > detail::deduce_numeric_epsilon<float>() * 10) {
                                    spg(tagDstOffset + d, cellno) /= wd + 0.0001;
                                }
                            }
                        });
            else
                cudaPol(range(spg.numBlocks() * spg.block_size),
                        [spg = proxy<space>(spg), tagDstOffset = spg._grid.getPropertyOffset(tag),
                         wOffset = spg._grid.getPropertyOffset("weight"),
                         nchns] __device__(std::size_t cellno) mutable {
                            auto w = spg(wOffset, cellno);
                            if (w > detail::deduce_numeric_epsilon<float>() * 10) {
                                for (int d = 0; d < nchns; ++d)
                                    spg(tagDstOffset + d, cellno) /= w + 0.0001;
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
                                         {gParamType_String, "ParticleAttribute", ""},
                                         {gParamType_String, "GridAttribute"},
                                         {"enum replace-all replace-local accumulate", "OpType", "replace-all"},
                                         {"enum linear quadratic cubic delta2 delta3 delta4", "Kernel", "quadratic"},
                                         {gParamType_Bool, "staggered", "0"},
                                         {gParamType_Bool, "initialize", "1"},
                                         {gParamType_Bool, "normalize", "1"},
                                         {gParamType_Bool, "activateBlock", "0"},
                                         {gParamType_Bool, "activateMultiGrid", "0"}},
                                        /* outputs: */
                                        {"SparseGrid"},
                                        /* params: */
                                        {},
                                        /* category: */
                                        {"Eulerian"},
                                    });

} // namespace zeno