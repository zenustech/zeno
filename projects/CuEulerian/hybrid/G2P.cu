#include "Structures.hpp"
#include "Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
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

    template <zs::kernel_e knl>
    void g2p_staggered(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *zsSPG, ZenoParticles *parObjPtr,
                       zs::SmallString srcTag, zs::SmallString parTag, const int nchns, bool isAccumulate) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = zsSPG->getSparseGrid();
        auto &pars = parObjPtr->getParticles();

        pol(range(pars.size()),
            [spgv = proxy<space>(spg), pars = proxy<space>({}, pars), tagSrcOffset = spg.getPropertyOffset(srcTag),
             tagDstOffset = pars.getPropertyOffset(parTag), nchns, isAccumulate] __device__(std::size_t pi) mutable {
                auto pos = pars.pack(dim_c<3>, "x", pi);
                for (int d = 0; d < nchns; ++d) { // 0, 1, 2
                    auto arena = spgv.wArena(pos, d, wrapv<knl>{});

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
    }

    template <zs::kernel_e knl>
    void g2p(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *zsSPG, ZenoParticles *parObjPtr, zs::SmallString srcTag,
             zs::SmallString parTag, const int nchns, bool isAccumulate) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = zsSPG->getSparseGrid();
        auto &pars = parObjPtr->getParticles();

        pol(range(pars.size()),
            [spgv = proxy<space>(spg), pars = proxy<space>({}, pars), tagSrcOffset = spg.getPropertyOffset(srcTag),
             tagDstOffset = pars.getPropertyOffset(parTag), nchns, isAccumulate] __device__(std::size_t pi) mutable {
                auto pos = pars.pack(dim_c<3>, "x", pi);
                auto arena = spgv.wArena(pos, wrapv<knl>{});

                float val[3];
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

    void apply() override {
        auto zsSPG = get_input<ZenoSparseGrid>("SparseGrid");
        auto &spg = zsSPG->getSparseGrid();
        auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
        auto attrTag = get_input2<std::string>("GridAttribute");
        auto parTag = get_input2<std::string>("ParticleAttribute");
        auto opType = get_input2<std::string>("OpType");
        auto kernel = get_input2<std::string>("Kernel");
        bool isStaggered = get_input2<bool>("staggered");

        bool isAccumulate = false;
        if (opType == "accumulate")
            isAccumulate = true;

        auto tag = src_tag(zsSPG, attrTag);

        using namespace zs;
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

        const int nchns = spg.getPropertySize(tag);
        if (isStaggered && nchns != 3)
            throw std::runtime_error("the size of GridAttribute is not 3!");

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
                match([&](auto knlTag) {
                    constexpr auto knt = decltype(knlTag)::value;
                    g2p_staggered<knt>(cudaPol, zsSPG.get(), parObjPtr, tag, parTag, nchns, isAccumulate);
                })(knl);
            } else {
                if (nchns > 3)
                    throw std::runtime_error("# of chns of property cannot exceed 3.");

                match([&](auto knlTag) {
                    constexpr auto knt = decltype(knlTag)::value;
                    g2p<knt>(cudaPol, zsSPG.get(), parObjPtr, tag, parTag, nchns, isAccumulate);
                })(knl);
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
                                      {"enum linear quadratic cubic delta2 delta3 delta4", "Kernel", "quadratic"},
                                      {"bool", "staggered", "0"}},
                                     /* outputs: */
                                     {"ZSParticles"},
                                     /* params: */
                                     {},
                                     /* category: */
                                     {"Eulerian"}});

struct ZSSparseGridAsParticles : INode {
    zs::Vector<zs::vec<float, 3>> transform_spgblock_to_particles(zs::CudaExecutionPolicy &pol, ZenoSparseGrid *zsSPG) {
        using namespace zs;
        constexpr auto space = RM_CVREF_T(pol)::exec_tag::value;

        auto &spg = zsSPG->getSparseGrid();
        auto nbs = spg.numBlocks();

        zs::Vector<zs::vec<float, 3>> blockCenters(spg.get_allocator(), nbs);

        // fmt::print("spg memspace : {}\n", (int)spg.memspace());
        pol(range(nbs),
            [spgv = proxy<space>(spg), pars = proxy<space>(blockCenters)]__device__(size_t bno) mutable {
                auto bcoord = spgv.wCoord(bno, 0);
                pars[bno] = bcoord + spgv.side_length * spgv.voxelSize() / 2;
            });
        return blockCenters;
    }

    void apply() override {
        auto zsSPG = get_input<ZenoSparseGrid>("SparseGrid");
        auto &spg = zsSPG->getSparseGrid();

        using namespace zs;
        auto pol = cuda_exec();
        auto centers = transform_spgblock_to_particles(pol, zsSPG.get());
        centers = centers.clone({memsrc_e::host, -1});

        auto prim = std::make_shared<PrimitiveObject>();

        prim->resize(centers.size());
        std::memcpy(prim->verts.values.data(), centers.data(), sizeof(zeno::vec3f) * centers.size());
#if 0
        pol(zip(prim->verts.values, centers), [](auto &dst, const auto& src) {
            dst = zeno::vec3f{src[0], src[1], src[2]};
        });
#endif

        set_output("prim", prim);
    }
};

ZENDEFNODE(ZSSparseGridAsParticles, {/* inputs: */
                                     {"SparseGrid"},
                                     /* outputs: */
                                     {"prim"},
                                     /* params: */
                                     {},
                                     /* category: */
                                     {"Eulerian"}});



} // namespace zeno