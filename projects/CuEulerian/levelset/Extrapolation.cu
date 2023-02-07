#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/profile/CppTimers.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

#include <zeno/utils/log.h>

#include "../scheme.hpp"
#include "../utils.cuh"

namespace zeno {

struct ZSGridExtrapolateAttr : INode {
    void apply() override {
        auto zsGrid = get_input<ZenoSparseGrid>("Grid");
        auto attrTag = get_input2<std::string>("Attribute");
        auto isStaggered = get_input2<bool>("Staggered");
        auto sdfTag = get_input2<std::string>("SDFAttrName");
        auto direction = get_input2<std::string>("Direction");
        auto maxIter = get_input2<int>("Iterations");

        auto &spg = zsGrid->spg;

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        if (!spg.hasProperty(sdfTag))
            throw std::runtime_error(fmt::format("the SDFAttribute [{}] does not exist!", sdfTag));

        auto tag = src_tag(zsGrid, attrTag);
        int nchns = spg.getPropertySize(tag);
        if (isStaggered && nchns != 3)
            throw std::runtime_error("the size of the Attribute is not 3!");

        auto dx = spg.voxelSize()[0];
        auto dt = 0.5f * dx;

        // calculate normal vector
        // "adv" - normal, "tmp" - double buffer, included in NSGrid
        spg.append_channels(pol, {{"adv", 3}, {"tmp", 3}});
        pol(zs::Collapse{spg.numBlocks(), spg.block_size},
            [spgv = zs::proxy<space>(spg), sdfOffset = spg.getPropertyOffset(sdfTag),
             dx] __device__(int blockno, int cellno) mutable {
                auto icoord = spgv.iCoord(blockno, cellno);

                // To do: shared memory and Neumann condition
                float sdf_this = spgv.value(sdfOffset, blockno, cellno);
                float sdf_x[2], sdf_y[2], sdf_z[2];
                for (int i = -1; i <= 1; i += 2) {
                    int arrid = (i + 1) >> 1;
                    sdf_x[arrid] = spgv.hasVoxel(icoord + zs::vec<int, 3>(i, 0, 0))
                                       ? spgv.value(sdfOffset, icoord + zs::vec<int, 3>(i, 0, 0))
                                       : sdf_this;
                    sdf_y[arrid] = spgv.hasVoxel(icoord + zs::vec<int, 3>(0, i, 0))
                                       ? spgv.value(sdfOffset, icoord + zs::vec<int, 3>(0, i, 0))
                                       : sdf_this;
                    sdf_z[arrid] = spgv.hasVoxel(icoord + zs::vec<int, 3>(0, 0, i))
                                       ? spgv.value(sdfOffset, icoord + zs::vec<int, 3>(0, 0, i))
                                       : sdf_this;
                }

                zs::vec<float, 3> normal;
                normal[0] = (sdf_x[1] - sdf_x[0]) / (2.f * dx);
                normal[1] = (sdf_y[1] - sdf_y[0]) / (2.f * dx);
                normal[2] = (sdf_z[1] - sdf_z[0]) / (2.f * dx);

                normal /= zs::max(normal.length(), zs::limits<float>::epsilon() * 10);

                spgv._grid.tuple(zs::dim_c<3>, "adv", blockno * spgv.block_size + cellno) = normal;
            });

        zs::SmallString tagSrc{tag}, tagDst{"tmp"};
        pol(zs::Collapse{spg.numBlocks(), spg.block_size},
            [spgv = zs::proxy<space>(spg), nchns, tagSrcOffset = spg.getPropertyOffset(tagSrc),
             tagDstOffset = spg.getPropertyOffset(tagDst)] __device__(int blockno, int cellno) mutable {
                for (int ch = 0; ch < nchns; ++ch) {
                    spgv(tagDstOffset + ch, blockno, cellno) = spgv.value(tagSrcOffset + ch, blockno, cellno);
                }
            });

        for (int iter = 0; iter < maxIter; ++iter) {
            if (iter % 2 == 0) {
                tagSrc = tag;
                tagDst = "tmp";
            } else {
                tagSrc = "tmp";
                tagDst = tag;
            }

            pol(zs::Collapse{spg.numBlocks(), spg.block_size},
                [spgv = zs::proxy<space>(spg), dx, dt, extDir = zs::SmallString{direction}, nchns, isStaggered,
                 tagSrcOffset = spg.getPropertyOffset(tagSrc), tagDstOffset = spg.getPropertyOffset(tagDst),
                 sdfOffset = spg.getPropertyOffset(sdfTag)] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
                    float sdf = spgv.value(sdfOffset, blockno, cellno);
                    zs::vec<float, 3> normal = spgv._grid.pack(zs::dim_c<3>, "adv", blockno * spgv.block_size + cellno);

                    for (int ch = 0; ch < nchns; ++ch) {
                        auto wcoord_face = spgv.wStaggeredCoord(blockno, cellno, ch);
                        if (isStaggered) {
                            sdf = spgv.wSample(sdfOffset, wcoord_face);
                        }

                        if ((extDir == "negative" && sdf < -zs::limits<float>::epsilon() * 10) ||
                            (extDir == "positive" && sdf > zs::limits<float>::epsilon() * 10) || extDir == "both") {

                            if (isStaggered) {
                                normal[0] = spgv.wSample("adv", 0, wcoord_face);
                                normal[1] = spgv.wSample("adv", 1, wcoord_face);
                                normal[2] = spgv.wSample("adv", 2, wcoord_face);

                                normal /= zs::max(normal.length(), zs::limits<float>::epsilon() * 10);
                            }

                            auto sign = [](float val) { return val > 0 ? 1 : -1; };

                            if (extDir == "both")
                                normal *= sign(sdf);
                            else if (extDir == "negative")
                                normal *= -1;

                            float v_self = spgv.value(tagSrcOffset + ch, blockno, cellno);
                            float v[3][2];
                            for (int i = -1; i <= 1; i += 2) {
                                int arrid = (i + 1) >> 1;
                                v[0][arrid] = spgv.value(tagSrcOffset + ch, icoord + zs::vec<int, 3>(i, 0, 0));
                                v[1][arrid] = spgv.value(tagSrcOffset + ch, icoord + zs::vec<int, 3>(0, i, 0));
                                v[2][arrid] = spgv.value(tagSrcOffset + ch, icoord + zs::vec<int, 3>(0, 0, i));
                            }

                            float adv_term = 0;
                            for (int dim = 0; dim < 3; ++dim) {
                                adv_term +=
                                    normal[dim] * scheme::upwind_1st(v[dim][0], v_self, v[dim][1], normal[dim], dx);
                            }

                            spgv(tagDstOffset + ch, blockno, cellno) = v_self - adv_term * dt;
                        }
                    }
                });
        }

        if (tagSrc == "tmp") {
            pol(zs::Collapse{spg.numBlocks(), spg.block_size},
                [spgv = zs::proxy<space>(spg), nchns, tagSrcOffset = spg.getPropertyOffset("tmp"),
                 tagDstOffset = spg.getPropertyOffset(tag)] __device__(int blockno, int cellno) mutable {
                    for (int ch = 0; ch < nchns; ++ch) {
                        spgv(tagDstOffset + ch, blockno, cellno) = spgv.value(tagSrcOffset + ch, blockno, cellno);
                    }
                });
        }

        set_output("Grid", zsGrid);
    }
};

ZENDEFNODE(ZSGridExtrapolateAttr, {/* inputs: */
                                   {"Grid",
                                    {"string", "Attribute", ""},
                                    {"bool", "Staggered", "0"},
                                    {"string", "SDFAttrName", "sdf"},
                                    {"enum both positive negative", "Direction", "positive"},
                                    {"int", "Iterations", "5"}},
                                   /* outputs: */
                                   {"Grid"},
                                   /* params: */
                                   {},
                                   /* category: */
                                   {"Eulerian"}});

} // namespace zeno