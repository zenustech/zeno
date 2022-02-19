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
              if (auto blockno = ls._table.insert(blockid);
                  blockno !=
                  table_t::sentinel_v) { // initialize newly inserted block
                auto block = ls._grid.block(blockno);
                for (typename ls_t::channel_counter_type chn = 0;
                     chn != ls.numChannels(); ++chn)
                  for (typename ls_t::cell_index_type ci = 0;
                       ci != ls.block_size; ++ci)
                    block(chn, ci) = ls._backgroundValue;
              }
            });
    fmt::print("levelset of [{}] blocks inserted [{}] blocks, eventually [{}] "
               "blocks\n",
               numPrevBlocks, numInBlocks, ls.numBlocks());
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
          if constexpr (true) {
            lsPtr->_i2wT = refLsPtr->_i2wT;
            lsPtr->_i2wRinv = refLsPtr->_i2wRinv;
            lsPtr->_i2wSinv = refLsPtr->_i2wSinv;
            lsPtr->_i2wRhat = refLsPtr->_i2wRhat;
            lsPtr->_i2wShat = refLsPtr->_i2wShat;
            lsPtr->_grid.dx = refLsPtr->_grid.dx;
          }
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
        {(std::size_t)ls.numBlocks(), (std::size_t)ls.block_size},
        [ls = proxy<execspace_e::cuda>(ls),
         refLs = proxy<execspace_e::cuda>(refLs),
         tag] __device__(typename RM_CVREF_T(ls)::size_type bi,
                         typename RM_CVREF_T(ls)::cell_index_type ci) mutable {
          using ls_t = RM_CVREF_T(ls);
          using grid_t = RM_CVREF_T(ls._grid);
          using vec3 = zs::vec<float, 3>;
#if 0
          if (ls.hasProperty("mark")) {
            if ((int)ls._grid("mark", bi, ci) == 0)
              return; // skip inactive voxels
          }
#endif
          auto blockid = ls._table._activeKeys[bi];
          auto cellid = ls_t::cellid_to_coord(ci);
          const auto propOffset = ls.propertyOffset(tag.name);
          const auto refPropOffset = refLs.propertyOffset(tag.name);
          if constexpr (ls_t::category == grid_e::staggered) {
            zs::vec<typename ls_t::TV, 3> Xs{
                refLs.worldToIndex(ls.indexToWorld(blockid + cellid, 0)),
                refLs.worldToIndex(ls.indexToWorld(blockid + cellid, 1)),
                refLs.worldToIndex(ls.indexToWorld(blockid + cellid, 2))};
            for (typename ls_t::channel_counter_type chn = 0;
                 chn != tag.numChannels; ++chn)
              ls._grid(propOffset + chn, bi, ci) =
                  refLs.isample(refPropOffset + chn, Xs[chn % 3], 0);
          } else {
            auto x = ls.indexToWorld(blockid + cellid);
            auto X = refLs.worldToIndex(x);
            // not quite efficient
            for (typename ls_t::channel_counter_type chn = 0;
                 chn != tag.numChannels; ++chn)
              ls._grid(propOffset + chn, bi, ci) =
                  refLs.isample(refPropOffset + chn, X, refLs._backgroundValue);
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