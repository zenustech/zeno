#include "../Structures.hpp"
#include "../Utils.hpp"
#include "zensim/geometry/LevelSetUtils.tpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/simulation/Utils.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>

#include "zensim/Logger.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetAdvect.h>
#include <openvdb/tools/VolumeAdvect.h>
#include <zeno/VDBGrid.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/zeno.h>

namespace zeno {

/// binary operation
struct WriteZSLevelSetToVDBGrid : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing WriteZSLevelSetToVDBGrid\n");
    auto vdb = get_input<VDBFloatGrid>("vdbGrid");

    // openvdb::FloatGrid::Ptr gp;
    if (has_input<ZenoLevelSet>("ZSLevelSet")) {
      auto ls = get_input<ZenoLevelSet>("ZSLevelSet");
      if (ls->holdsBasicLevelSet()) {
        zs::match(
            [&vdb](auto &lsPtr)
                -> std::enable_if_t<
                    zs::is_spls_v<typename RM_CVREF_T(lsPtr)::element_type>> {
              using LsT = typename RM_CVREF_T(lsPtr)::element_type;
              vdb->m_grid = zs::convert_sparse_levelset_to_vdbgrid(*lsPtr)
                                .template as<openvdb::FloatGrid::Ptr>();
            },
            [](...) {})(ls->getBasicLevelSet()._ls);
      } else
        ZS_WARN("The current input levelset is not a sparse levelset!");
    }

    fmt::print(fg(fmt::color::cyan),
               "done executing WriteZSLevelSetToVDBGrid\n");
    set_output("vdbGrid", std::move(vdb));
  }
};

ZENDEFNODE(WriteZSLevelSetToVDBGrid, {
                                         {
                                             {"ZSLevelSet", "vdbGrid"},
                                             {"string", "Attr", "sdf"},
                                         },
                                         {"vdbGrid"},
                                         {},
                                         {"Volume"},
                                     });

} // namespace zeno