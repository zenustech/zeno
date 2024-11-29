#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/SparseGrid.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"

#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

// #include <zeno/VDBGrid.h>

#include "../utils.cuh"
#include "zeno/utils/log.h"

namespace zeno {

struct ZSSParseGridDifference : INode {
  void apply() override {
    using namespace zs;
    auto grid = get_input<ZenoSparseGrid>("ZSGrid");

    auto attrTag = get_input2<std::string>("attrName");
    auto chnOffset = get_input2<int>("channelOffset");

    auto outputAttrTag = get_input2<std::string>("outputAttrName");
    if (outputAttrTag.empty())
      throw std::runtime_error(
          "[outputAttrName] should not be an empty string.");

    auto orientationStr = get_input2<std::string>("orientation");
    int orientation =
        orientationStr == "ddx" ? 0 : (orientationStr == "ddy" ? 1 : 2);

    auto boundaryStr = get_input2<std::string>("boundary_type");
    int boundaryType = boundaryStr == "neumann" ? 0 : /*dirichlet*/ 1;

    auto &spg = grid->spg;
    auto block_cnt = spg.numBlocks();
    auto pol = cuda_exec();
    constexpr auto space = execspace_e::cuda;

    spg.append_channels(pol, {{outputAttrTag, 1}});

    pol(Collapse{block_cnt, spg.block_size},
        [spgv = proxy<space>(spg),
         srcOffset = spg.getPropertyOffset(attrTag) + chnOffset, orientation,
         dstOffset = spg.getPropertyOffset(outputAttrTag), boundaryType,
         twodx = 2 * spg.voxelSize()[0]] __device__(int blockno,
                                                    int cellno) mutable {
          auto icoord = spgv.iCoord(blockno, cellno);
          auto val = spgv(srcOffset, blockno, cellno);
          auto iCoordA = icoord;
          iCoordA[orientation]++;
          auto iCoordB = icoord;
          iCoordB[orientation]--;

          auto getVal = [&](const auto &coord) -> zs::f32 {
            auto [bno, cno] = spgv.decomposeCoord(coord);
            if (bno == spgv.sentinel_v) {
              // boundary
              if (boundaryType == 0) // neumann
                return val;
              else
                return 0.f;
            } else {
              return spgv(srcOffset, bno, cno);
            }
          };
          auto tmp = (getVal(iCoordA) - getVal(iCoordB)) / twodx;
#if 0
          if (zs::abs(tmp) > 0.01) {
            printf("coord (%d, %d, %d) - (%d, %d, %d) diff: %f\n", iCoordA[0],
                   iCoordA[1], iCoordA[2], iCoordB[0], iCoordB[1], iCoordB[2],
                   (float)tmp);
          }
#endif
          spgv(dstOffset, blockno, cellno) = tmp;
        });

    set_output("ZSGrid", grid);
  }
};

ZENDEFNODE(ZSSParseGridDifference,
           {/* inputs: */
            {
                "ZSGrid",
                {"string", "attrName", "sdf"},
                {"int", "channelOffset", "0"},
                {"enum ddx ddy ddz", "orientation", "ddx"},
                {"string", "outputAttrName", ""},
                {"enum neumann dirichlet", "boundary_type", "neumann"},
            },
            /* outputs: */
            {"ZSGrid"},
            /* params: */
            {},
            /* category: */
            {"Eulerian"}});

} // namespace zeno