#include "../Structures.hpp"
#include "../Utils.hpp"
#include "zensim/geometry/LevelSetUtils.tpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
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
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing MarkZSLevelSet\n");

    using namespace zs;

    // this could possibly be the same staggered velocity field too
    auto zsfield = get_input<ZenoLevelSet>("ZSField");
    auto &field = zsfield->getBasicLevelSet()._ls;

    auto threshold = std::abs(get_input2<float>("threshold"));

    match(
        [this, threshold](auto &lsPtr)
            -> std::enable_if_t<
                is_spls_v<typename RM_CVREF_T(lsPtr)::element_type>> {
          // using SplsT = typename RM_CVREF_T(lsPtr)::element_type;
          auto cudaPol = cuda_exec().device(0);
          mark_level_set(cudaPol, *lsPtr, threshold);
          // refit_level_set_domain(cudaPol, *lsPtr, threshold);
        },
        [](...) { throw std::runtime_error("field is not spls!"); })(field);

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

/// match topology (partition union, ignore transformation difference)
struct ZSLevelSetTopologyUnion : INode {
  template <typename SplsT, typename TableT>
  void topologyUnion(SplsT &ls, const TableT &refTable) {
    using namespace zs;
    auto cudaPol = cuda_exec().device(0);

    const auto numInBlocks = refTable.size();
    const auto numPrevBlocks = ls.numBlocks();
    cudaPol(range(numInBlocks),
            [ls = proxy<execspace_e::cuda>(ls),
             refTable = proxy<execspace_e::cuda>(
                 refTable)] __device__(typename RM_CVREF_T(ls)::size_type
                                           bi) mutable {
              using ls_t = RM_CVREF_T(ls);
              using table_t = RM_CVREF_T(refTable);
              auto blockid = refTable._activeKeys[bi];
              ls._table.insert(blockid);
            });
    const auto numCurBlocks = ls.numBlocks();
    cudaPol(Collapse{numCurBlocks - numPrevBlocks, ls.block_size},
            [numPrevBlocks, ls = proxy<execspace_e::cuda>(ls)] __device__(
                auto bi, auto ci) mutable {
              using ls_t = RM_CVREF_T(ls);
              auto block = ls._grid.block(bi + numPrevBlocks);
              for (typename ls_t::channel_counter_type chn = 0;
                   chn != ls.numChannels(); ++chn)
                block(chn, ci) = 0; // ls._backgroundValue;
            });
    fmt::print("levelset of [{}] blocks inserted [{}] blocks, eventually [{}] "
               "blocks\n",
               numPrevBlocks, numInBlocks, numCurBlocks);
  }
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing ZSLevelSetTopologyUnion\n");

    using namespace zs;

    // this could possibly be the same staggered velocity field too
    auto zsfield = get_input<ZenoLevelSet>("ZSField");
    auto &field = zsfield->getBasicLevelSet()._ls;
    auto refZsField = get_input<ZenoLevelSet>("RefZSField");
    auto &refField = refZsField->getBasicLevelSet()._ls;

    match(
        [this](auto &lsPtr, const auto &refLsPtr)
            -> std::enable_if_t<
                is_spls_v<typename RM_CVREF_T(lsPtr)::element_type> &&
                is_spls_v<typename RM_CVREF_T(refLsPtr)::element_type>> {
          auto &table = lsPtr->_table;
          const auto &refTable = refLsPtr->_table;
          auto numBlocks = lsPtr->numBlocks();
          auto numRefBlocks = refLsPtr->numBlocks();
          auto cudaPol = cuda_exec().device(0);
          /// reserve enough memory
          lsPtr->resize(cudaPol, numBlocks + numRefBlocks);
          /// assume sharing the same transformation
          lsPtr->_backgroundValue = refLsPtr->_backgroundValue;
          lsPtr->_backgroundVecValue = refLsPtr->_backgroundVecValue;

          using TV = RM_CVREF_T(lsPtr->_min);
          lsPtr->_min = TV::init([&a = lsPtr->_min, &b = refLsPtr->_min](
                                     int i) { return std::min(a(i), b(i)); });
          lsPtr->_max = TV::init([&a = lsPtr->_max, &b = refLsPtr->_max](
                                     int i) { return std::max(a(i), b(i)); });
          lsPtr->_i2wT = refLsPtr->_i2wT;
          lsPtr->_i2wRinv = refLsPtr->_i2wRinv;
          lsPtr->_i2wSinv = refLsPtr->_i2wSinv;
          lsPtr->_i2wRhat = refLsPtr->_i2wRhat;
          lsPtr->_i2wShat = refLsPtr->_i2wShat;
          lsPtr->_grid.dx = refLsPtr->_grid.dx; // don't forget this.
          topologyUnion(*lsPtr, refTable);
        },
        [](...) { throw std::runtime_error("not both fields are spls!"); })(
        field, refField);
    fmt::print(fg(fmt::color::cyan),
               "done executing ZSLevelSetTopologyUnion\n");
    set_output("ZSField", std::move(zsfield));
  }
};

ZENDEFNODE(ZSLevelSetTopologyUnion, {
                                        {"ZSField", "RefZSField"},
                                        {"ZSField"},
                                        {},
                                        {"Volume"},
                                    });

/// extend domain (by vel field)
struct ExtendZSLevelSet : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ExtendZSLevelSet\n");

    using namespace zs;

    // this could possibly be the same staggered velocity field too
    auto zsfield = get_input<ZenoLevelSet>("ZSField");
    auto &field = zsfield->getBasicLevelSet()._ls;
    auto velZsField = get_input<ZenoLevelSet>("ZSVelField");
    const auto &velField = velZsField->getBasicLevelSet()._ls;
    auto dt = get_input2<float>("dt");

    match(
        [this, dt](auto &lsPtr, const auto &velLsPtr)
            -> std::enable_if_t<
                is_spls_v<typename RM_CVREF_T(lsPtr)::element_type> &&
                is_spls_v<typename RM_CVREF_T(velLsPtr)::element_type>> {
          auto cudaPol = cuda_exec().device(0);
          auto vm = get_level_set_max_speed(cudaPol, *velLsPtr);
          int nvoxels = (int)std::ceil(vm * dt / lsPtr->_grid.dx);
          auto nlayers = std::max(
              (nvoxels + lsPtr->side_length - 1) / lsPtr->side_length, 2);
          fmt::print("before expanding: {} blocks, max vel: {}, covering {} "
                     "voxels, {} blocks\n",
                     lsPtr->numBlocks(), vm, nvoxels, nlayers);
          // mark_level_set(cudaPol, *lsPtr);
          refit_level_set_domain(cudaPol, *lsPtr, 1e-6);
          extend_level_set_domain(cudaPol, *lsPtr, nlayers);
        },
        [](...) { throw std::runtime_error("not both fields are spls!"); })(
        field, velField);
    fmt::print(fg(fmt::color::cyan), "done executing ExtendZSLevelSet\n");
    set_output("ZSField", std::move(zsfield));
  }
};

ZENDEFNODE(ExtendZSLevelSet,
           {
               {"ZSField", "ZSVelField", {"float", "dt", "0.1"}},
               {"ZSField"},
               {},
               {"Volume"},
           });

struct ZSLevelSetFloodFill : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ZSLevelSetFloodFill\n");

    auto zsfield = get_input<ZenoLevelSet>("ZSField");
    auto &field = zsfield->getBasicLevelSet()._ls;
    auto sdfZsField = get_input<ZenoLevelSet>("ZSSdfField");

    using namespace zs;

    auto cudaPol = cuda_exec().device(0);

    using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
    using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
    using const_transition_ls_t = typename ZenoLevelSet::const_transition_ls_t;
    match(
        [&](auto &lsPtr, const auto &sdfLs)
            -> std::enable_if_t<
                is_spls_v<typename RM_CVREF_T(lsPtr)::element_type>> {
          using ls_t = RM_CVREF_T(*lsPtr);
          using sdf_ls_t = RM_CVREF_T(sdfLs);
          if constexpr (is_same_v<sdf_ls_t, basic_ls_t>) {
            match([&cudaPol, &lsPtr](const auto &sdfLsPtr) {
              extend_level_set_domain(
                  cudaPol, *lsPtr,
                  get_level_set_view<execspace_e::cuda>(sdfLsPtr));
            })(sdfLs._ls);
          } else if constexpr (is_same_v<sdf_ls_t, const_sdf_vel_ls_t>) {
            match([&cudaPol, &lsPtr](auto lsv) {
              extend_level_set_domain(cudaPol, *lsPtr, SdfVelFieldView{lsv});
            })(sdfLs.template getView<execspace_e::cuda>());
          } else if constexpr (is_same_v<sdf_ls_t, const_transition_ls_t>) {
            match([&cudaPol, &lsPtr, &sdfLs](auto fieldPair) {
              auto &fvSrc = std::get<0>(fieldPair);
              auto &fvDst = std::get<1>(fieldPair);
              extend_level_set_domain(
                  cudaPol, *lsPtr,
                  TransitionLevelSetView{SdfVelFieldView{fvSrc},
                                         SdfVelFieldView{fvDst}, sdfLs._stepDt,
                                         sdfLs._alpha});
            })(sdfLs.template getView<zs::execspace_e::cuda>());
          }
        },
        [](auto &lsPtr, ...) {
          throw std::runtime_error(fmt::format("levelset [{}] is not a spls!",
                                               get_var_type_str(lsPtr)));
        })(field, sdfZsField->getLevelSet());

    fmt::print(fg(fmt::color::cyan), "done executing ZSLevelSetFloodFill\n");
    set_output("ZSField", zsfield);
  }
};

ZENDEFNODE(ZSLevelSetFloodFill, {
                                    {"ZSField", "ZSSdfField"},
                                    {"ZSField"},
                                    {},
                                    {"Volume"},
                                });

} // namespace zeno