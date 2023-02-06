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

        auto tag = src_tag(zsGrid, attrTag);
        int m_nchns = spg.getPropertySize(tag);
        if (isStaggered && m_nchns != 3)
            throw std::runtime_error("the size of the Attribute is not 3!");

        auto dx = spg.voxelSize()[0];

        spg.append_channels(pol, {{"tmp", 3}});
        // calculate normal vector
        pol(zs::Collapse{spg.numBlocks(), spg.block_size},
            [spgv = zs::proxy<space>(spg), sdfOffset = spg.getPropertyOffset(sdfTag),
             dx] __device__(int blockno, int cellno) mutable {
                auto icoord = spgv.iCoord(blockno, cellno);

                // To do: shared memory and Neumann condition
                float sdf_x[2], sdf_y[2], sdf_z[2];
                for (int i = -1; i <= 1; i += 2) {
                    int arrid = ++i >> 1;
                    sdf_x[arrid] = spgv(sdfOffset, icoord + zs::vec<int, 3>(i, 0, 0));
                    sdf_y[arrid] = spgv(sdfOffset, icoord + zs::vec<int, 3>(0, i, 0));
                    sdf_z[arrid] = spgv(sdfOffset, icoord + zs::vec<int, 3>(0, 0, i));
                }

                zs::vec<float, 3> normal;
                normal[0] = (sdf_x[1] - sdf_x[0]) / (2.f * dx);
                normal[1] = (sdf_y[1] - sdf_y[0]) / (2.f * dx);
                normal[2] = (sdf_z[1] - sdf_z[0]) / (2.f * dx);

                float nm_len = zs::max(zs::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]),
                                       zs::limits<float>::epsilon() * 10);

                normal /= nm_len;

                spgv._grid.tuple(zs::dim_c<3>, "tmp", blockno * spgv.block_size + cellno) = normal;
            });

        for (int iter = 0; iter < maxIter; ++iter) {
            pol(zs::Collapse{spg.numBlocks(), spg.block_size},
                [spgv = zs::proxy<space>(spg), tagOffset = spg.getPropertyOffset(tag),
                 sdfOffset = spg.getPropertyOffset(sdfTag)] __device__(int blockno, int cellno) mutable {
                    auto icoord = spgv.iCoord(blockno, cellno);
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
                                    {"enum both positive negative", "Direction", "both"},
                                    {"int", "Iterations", "3"}},
                                   /* outputs: */
                                   {"Grid"},
                                   /* params: */
                                   {},
                                   /* category: */
                                   {"Eulerian"}});

} // namespace zeno