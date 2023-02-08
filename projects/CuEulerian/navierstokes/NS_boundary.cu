#include "Structures.hpp"
#include "Utils.hpp"
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

struct ZSNSNaiveSolidWall : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto SolidSDFPtrs = RETRIEVE_OBJECT_PTRS(ZenoSparseGrid, "SolidSDF");
        auto SolidVelPtrs = RETRIEVE_OBJECT_PTRS(ZenoSparseGrid, "SolidVel");

        auto &spg = NSGrid->spg;
        auto block_cnt = spg.numBlocks();

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        for (auto &&[SolidSDF, SolidVel] : zs::zip(SolidSDFPtrs, SolidVelPtrs)) {
            auto &sdf = SolidSDF->spg;
            auto &vel = SolidVel->spg;

            pol(zs::Collapse{block_cnt, spg.block_size},
                [spgv = zs::proxy<space>(spg), sdfv = zs::proxy<space>(sdf), velv = zs::proxy<space>(vel),
                 vSrcTag = src_tag(NSGrid, "v")] __device__(int blockno, int cellno) mutable {
                    auto wcoord = spgv.wCoord(blockno, cellno);
                    auto solid_sdf = sdfv.wSample("sdf", wcoord);

                    if (solid_sdf < 0) {
                        auto vel_s = velv.wStaggeredPack("v", wcoord);
                        auto block = spgv.block(blockno);
                        block.template tuple<3>(vSrcTag, cellno) = vel_s;
                    }

                    spgv("sdf", blockno, cellno) = solid_sdf;
                });
        }

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSNSNaiveSolidWall, {/* inputs: */
                                {"NSGrid", "SolidSDF", "SolidVel"},
                                /* outputs: */
                                {"NSGrid"},
                                /* params: */
                                {},
                                /* category: */
                                {"Eulerian"}});

struct ZSNSCutCellWeight : INode {
    void apply() override {
        auto NSGrid = get_input<ZenoSparseGrid>("NSGrid");
        auto SolidSDF = get_input<ZenoSparseGrid>("SolidSDF");

        auto &sdf = SolidSDF->spg;
        auto &spg = NSGrid->spg;
        auto dx = spg.voxelSize()[0];

        auto pol = zs::cuda_exec();
        constexpr auto space = zs::execspace_e::cuda;

        // sample sdf to vertex
        pol(zs::Collapse{spg.numBlocks(), spg.block_size}, [spgv = zs::proxy<space>(spg), sdfv = zs::proxy<space>(sdf),
                                                            dx] __device__(int blockno, int cellno) mutable {
            auto wcoord = spgv.wCoord(blockno, cellno);
            auto solid_sdf = sdfv.wSample("sdf", wcoord - 0.5f * dx);

            spgv("sdf", blockno, cellno) = solid_sdf;
        });

        pol(zs::Collapse{spg.numBlocks(), spg.block_size}, [spgv = zs::proxy<space>(spg)] __device__(
                                                               int blockno, int cellno) mutable {
            auto icoord = spgv.iCoord(blockno, cellno);
            float ls[2][2][2];
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    for (int k = 0; k < 2; ++k) {
                        ls[i][j][k] = spgv.value("sdf", icoord + zs::vec<int, 3>(i, j, k));
                    }

            // x-face (look from positive x direction)
            spgv("cut", 0, blockno, cellno) = scheme::face_fraction(ls[0][0][0], ls[0][1][0], ls[0][0][1], ls[0][1][1]);

            // y-face (look from positive y direction)
            spgv("cut", 1, blockno, cellno) = scheme::face_fraction(ls[0][0][0], ls[0][0][1], ls[1][0][0], ls[1][0][1]);

            // z-face (look from positive z direction)
            spgv("cut", 2, blockno, cellno) = scheme::face_fraction(ls[0][0][0], ls[1][0][0], ls[0][1][0], ls[1][1][0]);
        });

        set_output("NSGrid", NSGrid);
    }
};

ZENDEFNODE(ZSNSCutCellWeight, {/* inputs: */
                               {"NSGrid", "SolidSDF"},
                               /* outputs: */
                               {"NSGrid"},
                               /* params: */
                               {},
                               /* category: */
                               {"Eulerian"}});

} // namespace zeno