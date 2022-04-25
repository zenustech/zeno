#include "../Structures.hpp"
#include "../Utils.hpp"
#include "zensim/geometry/LevelSetUtils.tpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/math/matrix/QRSVD.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/simulation/Utils.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

/// binary operation
struct ZSLevelSetBinaryOperator : INode {
  template <typename SplsT, typename Op,
            typename T = typename SplsT::value_type>
  void binaryOp(SplsT &lsa, const SplsT &lsb, T a, T b, Op op) {
    using namespace zs;
    auto cudaPol = cuda_exec().device(0);

    const auto numInBlocks = lsb.numBlocks();
    const auto numPrevBlocks = lsa.numBlocks();
    cudaPol(range(numInBlocks), [taba = proxy<execspace_e::cuda>(lsa._table),
                                 tabb = proxy<execspace_e::cuda>(
                                     lsb._table)] __device__(auto bi) mutable {
      auto blockid = tabb._activeKeys[bi];
      taba.insert(blockid);
    });
    const auto numCurBlocks = lsa.numBlocks();
    cudaPol(Collapse{numCurBlocks - numPrevBlocks, lsa.block_size},
            [numPrevBlocks, lsa = proxy<execspace_e::cuda>(lsa)] __device__(
                auto bi, auto ci) mutable {
              using ls_t = RM_CVREF_T(lsa);
              auto block = lsa._grid.block(bi + numPrevBlocks);
              for (typename ls_t::channel_counter_type chn = 0;
                   chn != lsa.numChannels(); ++chn)
                block(chn, ci) = 0; // ls._backgroundValue;
            });
    fmt::print("levelset of [{}] blocks inserted [{}] blocks, eventually [{}] "
               "blocks\n",
               numPrevBlocks, numInBlocks, numCurBlocks);
    cudaPol(Collapse{numCurBlocks, lsa.block_size},
            [lsa = proxy<execspace_e::cuda>(lsa),
             lsb = proxy<execspace_e::cuda>(lsb), a, b,
             op] __device__(auto bi, auto ci) mutable {
              using ls_t = RM_CVREF_T(lsa);
              auto coord = lsa._table._activeKeys[bi] +
                           ls_t::grid_view_t::cellid_to_coord(ci);
              auto [bnoB, cnoB] = lsb.decompose_coord(coord);
              auto block = lsa._grid.block(bi);
              for (typename ls_t::channel_counter_type propNo = 0;
                   propNo != lsa.numProperties(); ++propNo) {
                const auto &prop = lsa.getPropertyNames()[propNo];
                if (prop == "mark" || prop == "sdf")
                  continue; // skip property ["mark", "sdf"]
                const auto propIdB = lsb.propertyIndex(prop);
                if (propIdB == lsb.numProperties())
                  continue; // skip channels missing in lsb
                const auto propOffsetA = lsa.getPropertyOffsets()[propNo];
                const auto propOffsetB = lsb.propertyOffset(prop);
                const auto propSize = lsa.getPropertySizes()[propNo];
                for (typename ls_t::channel_counter_type chn = 0;
                     chn != propSize; ++chn) {
                  const auto &A = a * block(propOffsetA + chn, ci);
                  const auto &B =
                      b * lsb.value_or(propOffsetB + chn, bnoB, cnoB, 0);
                  block(propOffsetA + chn, ci) = op(a, b);
                }
              }
              for (typename ls_t::channel_counter_type chn = 0;
                   chn != lsa.numChannels(); ++chn)
                block(chn, ci) = 0; // ls._backgroundValue;
            });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing ZSLevelSetBinaryOperator\n");

    using namespace zs;

    // this could possibly be the same staggered velocity field too
    auto zsfielda = get_input<ZenoLevelSet>("ZSFieldA");
    auto &fielda = zsfielda->getBasicLevelSet()._ls;
    auto zsfieldb = get_input<ZenoLevelSet>("ZSFieldB");
    auto &fieldb = zsfieldb->getBasicLevelSet()._ls;

    float a = get_input2<float>("a");
    float b = get_input2<float>("b");
    auto opStr = get_param<std::string>("transfer");

    using Op = variant<static_plus, static_minus, static_multiplies,
                       static_divides<true>>;
    Op op{};
    if (opStr == "aX_plus_bY")
      op = static_plus{};
    else if (opStr == "aX_minus_bY")
      op = static_minus{};
    else if (opStr == "aX_mul_bY")
      op = static_multiplies{};
    else if (opStr == "aX_div_bY")
      op = static_divides<true>{};

    match(
        [this, a, b](auto &lsPtr, const auto &lsPtrB, auto op)
            -> std::enable_if_t<
                is_spls_v<typename RM_CVREF_T(lsPtr)::element_type> &&
                is_same_v<typename RM_CVREF_T(lsPtrB)::element_type,
                          typename RM_CVREF_T(lsPtr)::element_type>> {
          auto numBlocks = lsPtr->numBlocks();
          auto numBlocksB = lsPtrB->numBlocks();
          auto cudaPol = cuda_exec().device(0);
          /// reserve enough memory
          lsPtr->resize(cudaPol, numBlocks + numBlocksB);
          /// assume sharing the same transformation
          lsPtr->_backgroundValue = lsPtrB->_backgroundValue;
          lsPtr->_backgroundVecValue = lsPtrB->_backgroundVecValue;

          using TV = RM_CVREF_T(lsPtr->_min);
          lsPtr->_min = TV::init([&a = lsPtr->_min, &b = lsPtrB->_min](int i) {
            return std::min(a(i), b(i));
          });
          lsPtr->_max = TV::init([&a = lsPtr->_max, &b = lsPtrB->_max](int i) {
            return std::max(a(i), b(i));
          });
          lsPtr->_i2wT = lsPtrB->_i2wT;
          lsPtr->_i2wRinv = lsPtrB->_i2wRinv;
          lsPtr->_i2wSinv = lsPtrB->_i2wSinv;
          lsPtr->_i2wRhat = lsPtrB->_i2wRhat;
          lsPtr->_i2wShat = lsPtrB->_i2wShat;
          lsPtr->_grid.dx = lsPtrB->_grid.dx; // don't forget this.

          binaryOp(*lsPtr, *lsPtrB, a, b, op);
        },
        [](...) {
          throw std::runtime_error(
              "both fields are of different spls categories!");
        })(fielda, fieldb, op);
    fmt::print(fg(fmt::color::cyan),
               "done executing ZSLevelSetBinaryOperator\n");
    set_output("ZSFieldA", std::move(zsfielda));
  }
};

ZENDEFNODE(
    ZSLevelSetBinaryOperator,
    {
        {"ZSFieldA", "ZSFieldB", {"float", "a", "1"}, {"float", "b", "1"}},
        {"ZSFieldA"},
        {{"enum aX_plus_bY aX_minus_bY aX_mul_bY aX_div_bY", "operator",
          "aX_plus_bY"}},
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
          using refls_t = RM_CVREF_T(refLs);
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
            // zs::vec<typename ls_t::TV, 3> Xs{
            //    refLs.worldToIndex(ls.indexToWorld(coord, 0)),
            //    refLs.worldToIndex(ls.indexToWorld(coord, 1)),
            //    refLs.worldToIndex(ls.indexToWorld(coord, 2))};
            for (typename ls_t::channel_counter_type chn = 0;
                 chn != tag.numChannels; ++chn) {
              if constexpr (refls_t::category == grid_e::staggered) {
                auto arena = refLs.arena(ls.indexToWorld(coord, chn % 3),
                                         chn % 3, kernel_linear_c, wrapv<0>{});
                ls._grid(propOffset + chn, bi, ci) =
                    arena.isample(refPropOffset + chn, 0);
              } else {
                auto arena = refLs.arena(ls.indexToWorld(coord, chn % 3),
                                         kernel_linear_c, wrapv<0>{});
                ls._grid(propOffset + chn, bi, ci) =
                    arena.isample(refPropOffset + chn, 0);
              }
            }
            // ls._grid(propOffset + chn, bi, ci) =
            //    refLs.isample(refPropOffset + chn, Xs[chn % 3], 0);
          } else {
            auto x = ls.indexToWorld(coord);
            auto X = refLs.worldToIndex(x);
            // not quite efficient
            if constexpr (refls_t::category == grid_e::staggered) {
              for (typename ls_t::channel_counter_type chn = 0;
                   chn != tag.numChannels; ++chn) {
                auto arena = refLs.arena(X, chn % 3, kernel_linear_c,
                                         wrapv<0>{}, false_c);
                ls._grid(propOffset + chn, bi, ci) =
                    arena.isample(refPropOffset + chn, 0);
              }
              // ls._grid(propOffset + chn, bi, ci) =
              //    refLs.isample(refPropOffset + chn, X, 0);
            } else {
              auto arena = refLs.arena(X, kernel_linear_c, wrapv<0>{}, false_c);
              for (typename ls_t::channel_counter_type chn = 0;
                   chn != tag.numChannels; ++chn)
                ls._grid(propOffset + chn, bi, ci) =
                    arena.isample(refPropOffset + chn, 0);
            }
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

/// advection
// process: refit, extend, advect
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

struct ClampZSLevelSet : INode {
  template <typename SplsT, typename RefSplsT, typename VelSplsT>
  void clamp(SplsT &ls, const RefSplsT &refLs, const VelSplsT &velLs,
             const float dt) {
    using namespace zs;

    auto cudaPol = cuda_exec().device(0);

    /// ls & refLs better be of same category
    cudaPol(
        Collapse{ls.numBlocks(), ls.block_size},
        [ls = proxy<execspace_e::cuda>(ls),
         refLs = proxy<execspace_e::cuda>(refLs),
         velLs = proxy<execspace_e::cuda>(velLs),
         dt] __device__(auto bi, auto ci) mutable {
          using ls_t = RM_CVREF_T(ls);
          using ref_ls_t = RM_CVREF_T(refLs);
          auto coord = ls._table._activeKeys[bi] +
                       ls_t::grid_view_t::cellid_to_coord(ci);
          if constexpr (ls_t::category == grid_e::staggered) {
            for (int d = 0; d != 3; ++d) {
              auto x = ls.indexToWorld(coord, d);
              auto vi = velLs.getMaterialVelocity(x);
              auto xn = x - vi * dt;
              if constexpr (ref_ls_t::category == grid_e::staggered) {
                auto pad =
                    refLs.arena(xn, d, kernel_linear_c, wrapv<0>{}, true_c);
                auto mi = pad.minimum("vel", d);
                auto ma = pad.maximum("vel", d);
                auto v = ls._grid("vel", d, bi, ci);
                ls._grid("vel", d, bi, ci) = v < mi ? mi : (v > ma ? ma : v);
              } else {
                auto pad = refLs.arena(xn, kernel_linear_c, wrapv<0>{}, true_c);
                auto mi = pad.minimum("vel", d);
                auto ma = pad.maximum("vel", d);
                auto v = ls._grid("vel", d, bi, ci);
                ls._grid("vel", d, bi, ci) = v < mi ? mi : (v > ma ? ma : v);
              }
            }
          } else {
            auto x = ls.indexToWorld(coord);
            auto vi = velLs.getMaterialVelocity(x);
            auto xn = x - vi * dt;
            if constexpr (ref_ls_t::category == grid_e::staggered) {
              for (typename ls_t::channel_counter_type chn = 0;
                   chn != ls.numChannels(); ++chn) {
                int f = chn % 3;
                auto pad =
                    refLs.arena(xn, f, kernel_linear_c, wrapv<0>{}, true_c);
                auto mi = pad.minimum(chn);
                auto ma = pad.maximum(chn);
                auto v = ls._grid(chn, bi, ci);
                ls._grid(chn, bi, ci) = v < mi ? mi : (v > ma ? ma : v);
              }
            } else {
              auto pad = refLs.arena(xn, kernel_linear_c, wrapv<0>{}, true_c);
              for (typename ls_t::channel_counter_type chn = 0;
                   chn != ls.numChannels(); ++chn) {
                auto mi = pad.minimum(chn);
                auto ma = pad.maximum(chn);
                auto v = ls._grid(chn, bi, ci);
                ls._grid(chn, bi, ci) = v < mi ? mi : (v > ma ? ma : v);
              }
            }
          }
        });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ClampZSLevelSet\n");

    using namespace zs;

    // this could possibly be the same staggered velocity field too
    auto zsfield = get_input<ZenoLevelSet>("ZSField");
    auto &field = zsfield->getBasicLevelSet()._ls;

    auto zsreffield = get_input<ZenoLevelSet>("RefZSField");
    auto &reffield = zsreffield->getBasicLevelSet()._ls;

    auto velZsField = get_input<ZenoLevelSet>("ZSVelField");
    const auto &velField = velZsField->getBasicLevelSet()._ls;
    auto dt = get_input2<float>("dt");

    match(
        [this, dt](auto &lsPtr, auto &refLsPtr, const auto &velLsPtr)
            -> std::enable_if_t<
                is_spls_v<typename RM_CVREF_T(lsPtr)::element_type> &&
                is_spls_v<typename RM_CVREF_T(refLsPtr)::element_type> &&
                is_spls_v<typename RM_CVREF_T(velLsPtr)::element_type>> {
          clamp(*lsPtr, *refLsPtr, *velLsPtr, dt);
        },
        [](...) { throw std::runtime_error("not all fields are spls!"); })(
        field, reffield, velField);

    fmt::print(fg(fmt::color::cyan), "done executing ClampZSLevelSet\n");
    set_output("ZSField", std::move(zsfield));
  }
};

ZENDEFNODE(ClampZSLevelSet,
           {
               {"ZSField", "RefZSField", "ZSVelField", {"float", "dt", "0.1"}},
               {"ZSField"},
               {},
               {"Volume"},
           });

} // namespace zeno