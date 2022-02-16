#include "../Structures.hpp"
#include "../Utils.hpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/math/matrix/QRSVD.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/simulation/Utils.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

/// mark activity
struct MarkZSLevelSet : INode {
  template <typename SplsT>
  void mark(SplsT &ls, zs::Vector<zs::u64> &numActiveVoxels, float threshold) {
    using namespace zs;
    auto cudaPol = cuda_exec().device(0);
    ls.append_channels(cudaPol, {{"mark", 1}});

    cudaPol({(std::size_t)ls.numBlocks(), (std::size_t)ls.block_size},
            [ls = proxy<execspace_e::cuda>(ls), cnt = numActiveVoxels.data(),
             threshold] __device__(typename RM_CVREF_T(ls)::size_type bi,
                                   typename RM_CVREF_T(
                                       ls)::cell_index_type ci) mutable {
              using ls_t = RM_CVREF_T(ls);
              auto block = ls._grid.block(bi);
              // const auto nchns = ls.numChannels();
              for (typename ls_t::channel_counter_type propNo = 0;
                   propNo != ls.numProperties(); ++propNo) {
                if (ls.getPropertyNames()[propNo] == "mark")
                  continue; // skip property ["mark"]
                auto propOffset = ls.getPropertyOffsets()[propNo];
                auto propSize = ls.getPropertySizes()[propNo];
                for (typename ls_t::channel_counter_type chn = 0;
                     chn != propSize; ++chn) {
                  if (zs::abs(block(propOffset + chn, ci)) > threshold) {
                    block("mark", ci) = (u64)1;
                    atomic_add(exec_cuda, cnt, (u64)1);
                    break; // no need further checking
                  }
                }
              }
            });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing MarkZSLevelSet\n");

    using namespace zs;

    // this could possibly be the same staggered velocity field too
    auto zsfield = get_input<ZenoLevelSet>("ZSField");
    auto &field = zsfield->getBasicLevelSet()._ls;

    auto threshold = std::abs(get_input2<float>("threshold"));

    Vector<u64> numActiveVoxels{1, memsrc_e::device, 0};
    numActiveVoxels.setVal(0);
    match(
        [this, threshold, &numActiveVoxels](auto &lsPtr)
            -> std::enable_if_t<
                is_spls_v<typename RM_CVREF_T(lsPtr)::element_type>> {
          // using SplsT = typename RM_CVREF_T(lsPtr)::element_type;
          mark(*lsPtr, numActiveVoxels, threshold);
        },
        [](...) { throw std::runtime_error("field is not spls!"); })(field);
    auto n = numActiveVoxels.getVal();
    fmt::print("[{}] active voxels in total.\n", n);

    fmt::print(fg(fmt::color::cyan), "done executing MarkZSLevelSet\n");
    set_output("ZSField", std::move(zsfield));
  }
};

ZENDEFNODE(MarkZSLevelSet, {
                               {"ZSField", {"float", "threshold", "1e-6"}},
                               {"ZSField"},
                               {},
                               {"Volume"},
                           });
/// extend domain (by vel field)
/// match topology
/// binary operation
/// advection
struct AdvectZSLevelSet : INode {
  void apply() override {

    fmt::print(fg(fmt::color::green), "begin executing AdvectZSLevelSet\n");

    using namespace zs;

    // this could possibly be the same staggered velocity field too
    auto zsfield = get_input<ZenoLevelSet>("ZSField");
    auto &field = zsfield->getBasicLevelSet()._ls;
    auto dstZsField = std::make_shared<ZenoLevelSet>(*zsfield);
    auto &dstField = dstZsField->getBasicLevelSet()._ls;

    const auto &velField = get_input<ZenoLevelSet>("ZSVelField")
                               ->getSparseLevelSet<grid_e::staggered>();

    match(
        [velField, &dstField](const auto &lsPtr)
            -> std::enable_if_t<
                is_spls_v<typename RM_CVREF_T(lsPtr)::element_type>> {
          using SplsT = typename RM_CVREF_T(lsPtr)::element_type;
          const auto &srcLs = *lsPtr;
          auto &dstLs = *std::get<std::shared_ptr<SplsT>>(dstField);

          auto cudaPol = cuda_exec().device(0);

          if constexpr (SplsT::category == grid_e::staggered) {
            ;
          } else {
          }
        },
        [](...) { throw std::runtime_error("field is not spls!"); })(field);

    fmt::print(fg(fmt::color::cyan), "done executing AdvectZSLevelSet\n");
    set_output("ZSField", std::move(dstZsField));
  }
};

ZENDEFNODE(AdvectZSLevelSet,
           {
               {"ZSField", "ZSVelField", {"float", "dt", "0.1"}},
               {"ZSField"},
               {{"string", "path", ""}},
               {"Volume"},
           });

} // namespace zeno