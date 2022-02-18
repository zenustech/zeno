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
struct MatchZSLevelSetTopology : INode {
  template <typename SplsT, typename TableT>
  void topologyUnion(SplsT &ls, const TableT &refTable) {
    using namespace zs;
    auto cudaPol = cuda_exec().device(0);

    cudaPol(range(refTable.size()),
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
                    block(chn, ci) = 0;
              }
            });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing MatchZSLevelSetTopology\n");

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
          topologyUnion(*lsPtr, refTable);
        },
        [](...) { throw std::runtime_error("not both fields are spls!"); })(
        field, refField);
    fmt::print(fg(fmt::color::cyan),
               "done executing MatchZSLevelSetTopology\n");
    set_output("ZSField", std::move(zsfield));
  }
};

ZENDEFNODE(MatchZSLevelSetTopology, {
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

    cudaPol(
        {(std::size_t)ls.numBlocks(), (std::size_t)ls.block_size},
        [ls = proxy<execspace_e::cuda>(ls),
         refLs = proxy<execspace_e::cuda>(refLs),
         tag] __device__(typename RM_CVREF_T(ls)::size_type bi,
                         typename RM_CVREF_T(ls)::cell_index_type ci) mutable {
          using ls_t = RM_CVREF_T(ls);
          using grid_t = RM_CVREF_T(ls._grid);
          using vec3 = zs::vec<float, 3>;
          auto blockid = ls._table._activeKeys[bi];
          auto cellid = grid_traits<grid_t>::cellid_to_coord(ci);
          const auto propOffset = ls.propertyOffset(tag.name);
          const auto refPropOffset = refLs.propertyOffset(tag.name);
          if constexpr (ls_t::category == grid_e::staggered) {
            zs::vec<typename ls_t::TV, 3> Xs{
                refLs.worldToIndex(
                    ls.indexToWorld(blockid + cellid + vec3{0.f, 0.5f, 0.5f})),
                refLs.worldToIndex(
                    ls.indexToWorld(blockid + cellid + vec3{0.5f, 0.f, 0.5f})),
                refLs.worldToIndex(
                    ls.indexToWorld(blockid + cellid + vec3{0.5f, 0.5f, 0.f}))};
            for (typename ls_t::channel_counter_type chn = 0;
                 chn != tag.numChannels; ++chn)
              ls._grid(propOffset + chn, bi, ci) =
                  refLs.isample(refPropOffset + chn, Xs[chn % 3], 0);
          } else {
            auto X = refLs.worldToIndex(ls.indexToWorld(blockid + cellid));
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
    auto &refField = refZsField->getBasicLevelSet()._ls;
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
               {"ZSField", "RefZSField", {"float", "property", "sdf"}},
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