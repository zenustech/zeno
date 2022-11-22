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

struct ZSSparseGridToPrimitive : INode {
    void apply() override {
        auto zsSPG = get_input<ZenoSparseGrid>("SparseGrid");
        auto &spg = zsSPG->getSparseGrid();
        auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
        auto attrTag = get_input2<std::string>("GridAttribute");
        auto parTag = get_input2<std::string>("ParticleAttribute");
        auto opType = get_input2<std::string>("OpType");
        bool isStaggered = get_input2<bool>("staggered");

        bool isAccumulate = false;
        if (opType == "accumulate")
            isAccumulate = true;

        auto tag = src_tag(zsSPG, attrTag);

        using namespace zs;
        constexpr auto space = execspace_e::cuda;

        const int nchns = spg.getPropertySize(tag);
        if (isStaggered && nchns != 3)
            throw std::runtime_error("the size of GridAttribute is not 3!");

        auto cudaPol = cuda_exec().device(0);

        for (auto &&parObjPtr : parObjPtrs) {
            auto &pars = parObjPtr->getParticles();

            // check whether prim has the attribute
            if (!pars.hasProperty(parTag)) {
                pars.append_channels(cudaPol, {{parTag, nchns}});
            } else {
                int m_nchns = pars.getPropertySize(parTag);
                if (m_nchns != nchns)
                    throw std::runtime_error("the size of the ParticleAttribute doesn't match with GridAttribute");
            }

            if (isStaggered) {
                cudaPol(range(pars.size()),
                        [spgv = proxy<space>(spg), pars = proxy<space>({}, pars),
                         tagSrcOffset = spg.getPropertyOffset(tag), tagDstOffset = pars.getPropertyOffset(parTag),
                         isAccumulate, nchns] __device__(std::size_t pi) mutable {
                            auto pos = pars.pack(dim_c<3>, "x", pi);
                            for (int d = 0; d < nchns; ++d) { // 0, 1, 2
                                auto arena = spgv.wArena(pos, d, wrapv<kernel_e::quadratic>{});

                                float val = 0;
                                for (auto loc : ndrange<3>(RM_CVREF_T(arena)::width)) {
                                    auto coord = arena.coord(loc);
                                    auto [bno, cno] = spgv.decomposeCoord(coord);
                                    if (bno < 0) // skip non-exist voxels
                                        continue;
                                    auto W = arena.weight(loc);

                                    val += spgv(tagSrcOffset + d, bno, cno) * W;
                                }
                                if (isAccumulate) {
                                    pars(tagDstOffset + d, pi) += val;
                                } else {
                                    pars(tagDstOffset + d, pi) = val;
                                }
                            }
                        });
            } else {
                if (nchns > 144)
                    throw std::runtime_error("# of chns of property cannot exceed 144.");
                cudaPol(range(pars.size()),
                        [spgv = proxy<space>(spg), pars = proxy<space>({}, pars),
                         tagSrcOffset = spg.getPropertyOffset(tag), tagDstOffset = pars.getPropertyOffset(parTag),
                         isAccumulate, nchns] __device__(std::size_t pi) mutable {
                            auto pos = pars.pack(dim_c<3>, "x", pi);
                            auto arena = spgv.wArena(pos, wrapv<kernel_e::quadratic>{});

                            float val[144];
                            for (int d = 0; d < nchns; ++d)
                                val[d] = 0;

                            for (auto loc : arena.range()) {
                                auto coord = arena.coord(loc);
                                auto [bno, cno] = spgv.decomposeCoord(coord);
                                if (bno < 0) // skip non-exist voxels
                                    continue;

                                auto W = arena.weight(loc);
#pragma unroll
                                for (int d = 0; d < nchns; ++d) {
                                    val[d] += spgv(tagSrcOffset + d, bno, cno) * W;
                                }
                            }
                            for (int d = 0; d < nchns; ++d) {
                                if (isAccumulate) {
                                    pars(tagDstOffset + d, pi) += val[d];
                                } else {
                                    pars(tagDstOffset + d, pi) = val[d];
                                }
                            }
                        });
            }
        }

        set_output("ZSParticles", get_input("ZSParticles"));
    }
};

ZENDEFNODE(ZSSparseGridToPrimitive, {/* inputs: */
                                     {"SparseGrid",
                                      "ZSParticles",
                                      {"string", "GridAttribute"},
                                      {"string", "ParticleAttribute", ""},
                                      {"enum replace accumulate", "OpType", "replace"},
                                      {"bool", "staggered", "0"}},
                                     /* outputs: */
                                     {"ZSParticles"},
                                     /* params: */
                                     {},
                                     /* category: */
                                     {"Eulerian"}});
} // namespace zeno