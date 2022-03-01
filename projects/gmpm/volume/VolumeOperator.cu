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
/// resample
struct ResampleZSLevelSet : INode {
  template <typename SplsT, typename RefSplsT>
  void resample(SplsT &ls, const RefSplsT &refLs, const zs::PropertyTag &tag) {
    using namespace zs;

    auto cudaPol = cuda_exec().device(0);
    ls.append_channels(cudaPol, {tag}); // would also check channel dimension

    fmt::print(
        "tag: [{}, {}] at {} (out of {}) in dst, [{}] (out of {}) in ref.\n",
        tag.name, tag.numChannels, ls.getChannelOffset(tag.name),
        ls.numChannels(), refLs.getChannelOffset(tag.name),
        refLs.numChannels());
    cudaPol(
        Collapse{ls.numBlocks(), ls.block_size},
        [ls = proxy<execspace_e::cuda>(ls),
         refLs = proxy<execspace_e::cuda>(refLs),
         tag] __device__(typename RM_CVREF_T(ls)::size_type bi,
                         typename RM_CVREF_T(ls)::cell_index_type ci) mutable {
          using ls_t = RM_CVREF_T(ls);
          using vec3 = zs::vec<float, 3>;
#if 0
          if (ls.hasProperty("mark")) {
            if ((int)ls("mark", bi, ci) == 0)
              return; // skip inactive voxels
          }
#endif
          auto coord = ls._table._activeKeys[bi] +
                       ls_t::grid_view_t::cellid_to_coord(ci);
          const auto propOffset = ls.propertyOffset(tag.name);
          const auto refPropOffset = refLs.propertyOffset(tag.name);
          if constexpr (ls_t::category == grid_e::staggered) {
            zs::vec<typename ls_t::TV, 3> Xs{
                refLs.worldToIndex(ls.indexToWorld(coord, 0)),
                refLs.worldToIndex(ls.indexToWorld(coord, 1)),
                refLs.worldToIndex(ls.indexToWorld(coord, 2))};
            for (typename ls_t::channel_counter_type chn = 0;
                 chn != tag.numChannels; ++chn)
              ls._grid(propOffset + chn, bi, ci) =
                  refLs.isample(refPropOffset + chn, Xs[chn % 3], 0);
          } else {
            auto x = ls.indexToWorld(coord);
            auto X = refLs.worldToIndex(x);
            // not quite efficient
            for (typename ls_t::channel_counter_type chn = 0;
                 chn != tag.numChannels; ++chn)
              ls._grid(propOffset + chn, bi, ci) =
                  refLs.isample(refPropOffset + chn, X, 0);
          }
        });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ResampleZSLevelSet\n");

    using namespace zs;

    // this could possibly be the same staggered velocity field too
    auto zsfield = get_input<ZenoLevelSet>("ZSField");
    auto &field = zsfield->getBasicLevelSet()._ls;
    auto refZsField = get_input<ZenoLevelSet>("RefZSField");
    const auto &refField = refZsField->getBasicLevelSet()._ls;
    auto propertyName = get_input2<std::string>("property");

    match(
        [this, propertyName](auto &lsPtr, const auto &refLsPtr)
            -> std::enable_if_t<
                is_spls_v<typename RM_CVREF_T(lsPtr)::element_type> &&
                is_spls_v<typename RM_CVREF_T(refLsPtr)::element_type>> {
          PropertyTag tag{};
          tag.name = propertyName;
          tag.numChannels = refLsPtr->getChannelSize(propertyName);
          if (tag.numChannels == 0)
            throw std::runtime_error(fmt::format(
                "property [{}] not exists in the source field", propertyName));
          lsPtr->printTransformation("dst");
          refLsPtr->printTransformation("ref");
          resample(*lsPtr, *refLsPtr, tag);
        },
        [](...) { throw std::runtime_error("not both fields are spls!"); })(
        field, refField);
    fmt::print(fg(fmt::color::cyan), "done executing ResampleZSLevelSet\n");
    set_output("ZSField", std::move(zsfield));
  }
};

ZENDEFNODE(ResampleZSLevelSet,
           {
               {"ZSField", "RefZSField", {"string", "property", "sdf"}},
               {"ZSField"},
               {},
               {"Volume"},
           });
/// binary operation
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

/// advection
// process: shrink, extend, advect
// :
struct AdvectZSLevelSet : INode {
  template <typename SplsT, typename VelSplsT>
  int advect(SplsT &lsOut, const VelSplsT &velLs, const float dt) {
    using namespace zs;

    auto cudaPol = cuda_exec().device(0);

    auto vm = get_level_set_max_speed(cudaPol, velLs);
    int nvoxels = (int)std::ceil(vm * dt / lsOut._grid.dx);
    auto nlayers =
        std::max((nvoxels + lsOut.side_length - 1) / lsOut.side_length, 2);
    fmt::print("before expanding: {} blocks, max vel: {}, covering {} "
               "voxels, {} blocks\n",
               lsOut.numBlocks(), vm, nvoxels, nlayers);
    refit_level_set_domain(cudaPol, lsOut, 1e-6);
    extend_level_set_domain(cudaPol, lsOut, nlayers);

    const auto ls = lsOut.clone(lsOut.get_allocator());
    cudaPol(Collapse{lsOut.numBlocks(), lsOut.block_size},
            [ls = proxy<execspace_e::cuda>(ls),
             lsOut = proxy<execspace_e::cuda>(lsOut),
             velLs = proxy<execspace_e::cuda>(velLs),
             dt] __device__(typename RM_CVREF_T(lsOut)::size_type bi,
                            typename RM_CVREF_T(
                                lsOut)::cell_index_type ci) mutable {
              using ls_t = RM_CVREF_T(ls);
              using vel_ls_t = RM_CVREF_T(velLs);
              using vec3 = zs::vec<float, 3>;
              auto coord = lsOut._table._activeKeys[bi] +
                           ls_t::grid_view_t::cellid_to_coord(ci);
              if constexpr (ls_t::category == grid_e::staggered) {
                for (int d = 0; d != 3; ++d) {
                  auto x = lsOut.indexToWorld(coord, d);
                  auto vi = velLs.getMaterialVelocity(x);
                  auto xn = x - vi * dt;
                  auto Xn = ls.worldToIndex(xn);

                  lsOut._grid("vel", d, bi, ci) = ls.isample("vel", d, Xn, 0);
                }
              } else {
                auto x = lsOut.indexToWorld(coord);
                auto vi = velLs.getMaterialVelocity(x);
                auto xn = x - vi * dt;
                auto Xn = ls.worldToIndex(xn);
                for (typename ls_t::channel_counter_type chn = 0;
                     chn != ls.numChannels(); ++chn)
                  lsOut._grid(chn, bi, ci) = ls.isample(chn, Xn, 0);
              }
            });
    return nvoxels;
  }
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing AdvectZSLevelSet\n");

    using namespace zs;

    // this could possibly be the same staggered velocity field too
    auto zsfield = get_input<ZenoLevelSet>("ZSField");
    auto &field = zsfield->getBasicLevelSet()._ls;

    auto velZsField = get_input<ZenoLevelSet>("ZSVelField");
    const auto &velField = velZsField->getBasicLevelSet()._ls;
    auto dt = get_input2<float>("dt");

    auto nvoxels = std::make_shared<NumericObject>();

    match(
        [this, dt, &nvoxels](auto &lsPtr, const auto &velLsPtr)
            -> std::enable_if_t<
                is_spls_v<typename RM_CVREF_T(lsPtr)::element_type> &&
                is_spls_v<typename RM_CVREF_T(velLsPtr)::element_type>> {
          nvoxels->set(advect(*lsPtr, *velLsPtr, dt));
        },
        [](...) { throw std::runtime_error("not both fields are spls!"); })(
        field, velField);

    fmt::print(fg(fmt::color::cyan), "done executing AdvectZSLevelSet\n");
    set_output("ZSField", std::move(zsfield));
    set_output("nvoxels", std::move(nvoxels));
  }
};

ZENDEFNODE(AdvectZSLevelSet,
           {
               {"ZSField", "ZSVelField", {"float", "dt", "0.1"}},
               {"ZSField", "nvoxels"},
               {},
               {"Volume"},
           });

} // namespace zeno