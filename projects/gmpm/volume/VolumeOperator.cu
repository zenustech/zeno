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
            ;
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