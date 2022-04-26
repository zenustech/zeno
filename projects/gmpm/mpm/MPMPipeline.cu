#include "../Structures.hpp"
#include "../Utils.hpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/math/matrix/QRSVD.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/simulation/Utils.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

/// sparsity
struct ZSPartitionForZSParticles : INode {
  void apply() override {
    using namespace zs;
    fmt::print(fg(fmt::color::green),
               "begin executing ZSPartitionForZSParticles\n");
    auto table = get_input<ZenoPartition>("ZSPartition");
    auto &partition = table->get();
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();
    auto cudaPol = cuda_exec().device(0);

    bool cached = get_param<std::string>("strategy") == "cache" ? true : false;
    if (!table->requestRebuild && cached && table->hasTags()) {
      zs::Vector<int> bRebuild{1, memsrc_e::device, 0};
      bRebuild.setVal(0);
      cudaPol(range(table->numBoundaryEntries()), // table->getTags(),
              [tags = proxy<execspace_e::cuda>(table->getTags()),
               flag = proxy<execspace_e::cuda>(
                   bRebuild)] __device__(auto i) mutable {
                auto tag = tags[i];
                if (tag == 1 && flag[0] == 0) {
                  // atomic_cas(exec_cuda, &flag[0], 0, 1);
                  flag[0] = 1;
                }
              });
      // no boundary entry touched yet, no need for rebuild
      if (bRebuild.getVal() == 0) {
        table->rebuilt = false;
        fmt::print(fg(fmt::color::cyan),
                   "done executing ZSPartitionForZSParticles (skipping full "
                   "rebuild)\n");
        set_output("ZSPartition", table);
        return;
      }
    }

    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");

    std::size_t cnt = 0;
    for (auto &&parObjPtr : parObjPtrs) {
      cnt += (std::size_t)std::ceil(parObjPtr->getParticles().size() /
                                    get_input2<float>("ppb"));
      if (parObjPtr->isMeshPrimitive())
        cnt += (std::size_t)std::ceil(parObjPtr->numElements() /
                                      get_input2<float>("ppb"));
    }
    if (partition.size() * 2 < cnt)
      partition.resize(cudaPol, cnt * 2);

    using Partition = typename ZenoPartition::table_t;
    // reset
    partition.reset(cudaPol, true);

    using grid_t = typename ZenoGrid::grid_t;
    static_assert(grid_traits<grid_t>::is_power_of_two,
                  "grid side_length should be power of two");

    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();
      cudaPol(range(pars.size()),
              [pars = proxy<execspace_e::cuda>({}, pars),
               table = proxy<execspace_e::cuda>(partition),
               dxinv = 1.f / grid.dx] __device__(size_t pi) mutable {
                auto x = pars.template pack<3>("pos", pi);
                auto c = (x * dxinv - 0.5);
                typename Partition::key_t coord{};
                for (int d = 0; d != 3; ++d)
                  coord[d] = lower_trunc(c[d]);
                table.insert(coord - (coord & (grid_t::side_length - 1)));
              });
      if (parObjPtr->category != ZenoParticles::mpm) { // including tracker
        if (!parObjPtr->isMeshPrimitive())
          throw std::runtime_error(
              "The zsprimitive is not of mpm category but has no elements.");
        auto &eles = parObjPtr->getQuadraturePoints();
        cudaPol(range(eles.size()),
                [eles = proxy<execspace_e::cuda>({}, eles),
                 table = proxy<execspace_e::cuda>(partition),
                 dxinv = 1.f / grid.dx] __device__(size_t ei) mutable {
                  auto x = eles.template pack<3>("pos", ei);
                  auto c = (x * dxinv - 0.5);
                  typename Partition::key_t coord{};
                  for (int d = 0; d != 3; ++d)
                    coord[d] = lower_trunc(c[d]);
                  table.insert(coord - (coord & (grid_t::side_length - 1)));
                });
      }
    }
    if (cached) {
      table->reserveTags();
      identify_boundary_indices(cudaPol, *table, wrapv<grid_t::side_length>{});
    }
    table->rebuilt = true;
    if (table->requestRebuild) // request processed
      table->requestRebuild = false;

    fmt::print("partition of [{}] blocks for {} particles\n", partition.size(),
               cnt);

    fmt::print(fg(fmt::color::cyan),
               "done executing ZSPartitionForZSParticles\n");
    set_output("ZSPartition", table);
  }
};

ZENDEFNODE(ZSPartitionForZSParticles,
           {
               {"ZSPartition", "ZSGrid", "ZSParticles", {"float", "ppb", "1"}},
               {"ZSPartition"},
               {{"enum force cache", "strategy", "force"}},
               {"MPM"},
           });

struct ExpandZSPartition : INode {
  using grid_t = typename ZenoGrid::grid_t;

  template <typename VecT,
            zs::enable_if_t<std::is_integral_v<typename VecT::value_type>> = 0>
  static constexpr int getDirIndex(const zs::VecInterface<VecT> &dir) noexcept {
    // dir [-1, 1]
    auto offset = dir + 1;
    return offset[0] * 9 + offset[1] * 3 + offset[2];
  }

  void registerNewBlockEntries(zs::CudaExecutionPolicy &policy,
                               typename ZenoPartition::table_t &table,
                               zs::Vector<zs::i32> &dirTags, std::size_t offset,
                               std::size_t numNewEntries) const {
    using namespace zs;
    policy(range(numNewEntries), [table = proxy<execspace_e::cuda>(table),
                                  dirTags = proxy<execspace_e::cuda>(dirTags),
                                  offset] __device__(int bi) mutable {
      using table_t = RM_CVREF_T(table);
      bi += offset;
      auto bcoord = table._activeKeys[bi];
      using key_t = typename table_t::key_t;
      for (auto ijk : ndrange<3>(3)) {
        auto dir = make_vec<int>(ijk) - 1; // current expanding direction
        if (auto neighborNo = table.query(
                bcoord + dir * (int)grid_traits<grid_t>::side_length);
            neighborNo != table_t::sentinel_v)
          atomic_or(exec_cuda, &dirTags[neighborNo],
                    (i32)(1 << getDirIndex(-dir)));
      }
    });
  }
  void apply() override {
    using namespace zs;
    fmt::print(fg(fmt::color::green), "begin executing ExpandZSPartition\n");
    auto table = get_input<ZenoPartition>("ZSPartition");
    auto &partition = table->get();
    auto offset = get_param<int>("offset");
    auto extent = get_param<int>("extent");

    auto lower = std::abs(offset);
    auto higher = std::abs(offset + extent);
    auto niters = std::min(lower, higher);

    if (niters == 0 || !table->rebuilt) { // only expand after a fresh rebuilt
      fmt::print(fg(fmt::color::cyan), "done executing ExpandZSPartition "
                                       "(skipping expansion due to caching)\n");
      set_output("ZSPartition", std::move(table));
      return;
    }

    auto prevCnt = partition.size();
    fmt::print(
        "expect {} iterations to complete partition expansion of {} entries.\n",
        niters, prevCnt);

    // at least 27 bits for 3d[-1, 1] range
    Vector<i32> dirs{partition.get_allocator(), (std::size_t)prevCnt};
    dirs.reset(0);
    auto cudaPol = cuda_exec().device(0);
    registerNewBlockEntries(cudaPol, partition, dirs, 0, prevCnt);

    static_assert(grid_traits<grid_t>::is_power_of_two,
                  "grid side_length should be power of two");

    for (int iter = 0; iter != niters; ++iter) {
      cudaPol(range(prevCnt), [dirs = proxy<execspace_e::cuda>(dirs),
                               table = proxy<execspace_e::cuda>(
                                   partition)] __device__(auto bi) mutable {
        using table_t = RM_CVREF_T(table);
        auto blockid = table._activeKeys[bi];
        for (auto ijk : ndrange<3>(3)) {
          auto dir = make_vec<int>(ijk) - 1; // current expanding direction
          auto dirNo = getDirIndex(dir);
          if (dirs[bi] & (1 << dirNo)) // check if this direction is necessary
            continue;
          table.insert(blockid + dir * (int)grid_traits<grid_t>::side_length);
        }
      });
      auto curCnt = partition.size();
      fmt::print("partition insertion iter [{}]: [{}] blocks -> [{}] blocks\n",
                 iter, prevCnt, curCnt);

      dirs.resize(curCnt, 0);
      fmt::print("done dirtag resize\n");
      registerNewBlockEntries(cudaPol, partition, dirs, prevCnt,
                              curCnt - prevCnt);

      prevCnt = curCnt;
    }
    if (table->hasTags()) {
      // identify_boundary_indices(cudaPol, *table,
      // wrapv<grid_t::side_length>{});
      using Ti = ZenoPartition::Ti;
      using indices_t = ZenoPartition::indices_t;
      indices_t marks{partition.get_allocator(), (std::size_t)prevCnt + 1},
          offsets{partition.get_allocator(), (std::size_t)prevCnt + 1};
      cudaPol(Collapse{prevCnt}, [dirs = proxy<execspace_e::cuda>(dirs),
                                  marks = proxy<execspace_e::cuda>(
                                      marks)] __device__(Ti bi) mutable {
        marks[bi] = dirs[bi] != (i32)27;
      });
      exclusive_scan(cudaPol, std::begin(marks), std::end(marks),
                     std::begin(offsets));
      auto bouCnt = offsets.getVal(prevCnt);

      auto &boundaryIndices = table->getBoundaryIndices();
      boundaryIndices.resize(bouCnt);
      cudaPol(range(prevCnt),
              [marks = proxy<execspace_e::cuda>(marks),
               boundaryIndices = proxy<execspace_e::cuda>(boundaryIndices),
               offsets = proxy<execspace_e::cuda>(
                   offsets)] __device__(Ti bi) mutable {
                if (marks[bi])
                  boundaryIndices[offsets[bi]] = bi;
              });

      auto &tags = table->getTags();
      tags.resize(bouCnt);
      tags.reset(0);
    }

    fmt::print(fg(fmt::color::cyan), "done executing ExpandZSPartition\n");

    set_output("ZSPartition", std::move(table));
  }
};

ZENDEFNODE(ExpandZSPartition,
           {
               {"ZSPartition"},
               {"ZSPartition"},
               {{"int", "offset", "0"}, {"int", "extent", "2"}},
               {"MPM"},
           });

/// grid
struct ZSGridFromZSPartition : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing ZSGridFromZSPartition\n");
    auto zspartition = get_input<ZenoPartition>("ZSPartition");
    auto &partition = zspartition->get();
    auto cnt = partition.size();

    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    zsgrid->partition = zspartition;
    auto &grid = zsgrid->get();
    grid.resize(cnt);

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);
    // clear grid
    cudaPol(Collapse{cnt, ZenoGrid::grid_t::block_space()},
            [grid = proxy<execspace_e::cuda>({}, grid)] __device__(
                int bi, int ci) mutable {
              auto block = grid.block(bi);
              const auto nchns = grid.numChannels();
              for (int i = 0; i != nchns; ++i)
                block(i, ci) = 0;
            });

    fmt::print(fg(fmt::color::cyan), "done executing ZSGridFromZSPartition\n");
    set_output("ZSGrid", zsgrid);
  }
};

ZENDEFNODE(ZSGridFromZSPartition, {
                                      {"ZSPartition", "ZSGrid"},
                                      {"ZSGrid"},
                                      {},
                                      {"MPM"},
                                  });

struct UpdateZSGrid : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing UpdateZSGrid\n");
    // auto dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
    auto maxVelSqr = std::make_shared<NumericObject>();

    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();
    auto stepDt = get_input<NumericObject>("dt")->get<float>();

    using namespace zs;
    auto gravity = get_input2<float>("gravity");
    auto accel = zs::vec<float, 3>::zeros();
    if (has_input("Accel")) {
      auto tmp = get_input<NumericObject>("Accel")->get<vec3f>();
      accel = zs::vec<float, 3>{tmp[0], tmp[1], tmp[2]};
    } else
      accel[1] = gravity;

    Vector<float> velSqr{1, zs::memsrc_e::um, 0};
    velSqr[0] = 0;
    auto cudaPol = cuda_exec().device(0);

    if (zsgrid->isPicStyle())
      cudaPol(Collapse{partition.size(), ZenoGrid::grid_t::block_space()},
              [grid = proxy<execspace_e::cuda>({}, grid),
               /*table = proxy<execspace_e::cuda>(partition), */ stepDt, accel,
               ptr = velSqr.data()] __device__(auto bi, auto ci) mutable {
                auto block = grid.block(bi);
                auto mass = block("m", ci);
                if (mass != 0.f) {
                  mass = 1.f / mass;
                  auto vel = block.pack<3>("v", ci) * mass;
#if 0
                  if (vel.norm() > 0.2) {
                    auto pos =
                        (table._activeKeys[bi] + grid.cellid_to_coord(ci)) *
                        grid.dx;
                    printf("(%f, %f, %f) vel: %f, %f, %f\n", pos[0], pos[1],
                           pos[2], vel[0], vel[1], vel[2]);
                  }
#endif
                  vel += accel * stepDt;
                  block.set("v", ci, vel);
                  /// cfl dt
                  auto velSqr = vel.l2NormSqr();
                  atomic_max(exec_cuda, ptr, velSqr);
                }
              });
    else if (zsgrid->isFlipStyle())
      cudaPol(Collapse{partition.size(), ZenoGrid::grid_t::block_space()},
              [grid = proxy<execspace_e::cuda>({}, grid), stepDt, accel,
               ptr = velSqr.data()] __device__(auto bi, auto ci) mutable {
                auto block = grid.block(bi);
                auto mass = block("m", ci);
                if (mass != 0.f) {
                  mass = 1.f / mass;

                  auto vel = block.pack<3>("v", ci) * mass;
                  // vel += accel * stepDt;
                  block.set("v", ci, vel);

                  auto vstar =
                      block.pack<3>("vstar", ci) * mass + vel + accel * stepDt;
                  block.set("vstar", ci, vstar);

                  /// cfl dt
                  auto velSqr = vstar.l2NormSqr();
                  atomic_max(exec_cuda, ptr, velSqr);
                }
              });
    else if (zsgrid->transferScheme == "boundary")
      cudaPol(Collapse{partition.size(), ZenoGrid::grid_t::block_space()},
              [grid = proxy<execspace_e::cuda>({}, grid)] __device__(
                  auto bi, auto ci) mutable {
                auto block = grid.block(bi);
                auto mass = block("m", ci);
                if (mass != 0.f) {
                  {
                    mass = 1.f / mass;
                    auto vel = block.pack<3>("v", ci) * mass;
                    block.set("v", ci, vel);
                  }
                  auto nrm = block.pack<3>("nrm", ci) * mass;
                  block.set("nrm", ci, nrm.normalized());
                }
              });

#if 0
    cudaPol(Collapse{partition.size(), ZenoGrid::grid_t::block_space()},
            [grid = proxy<execspace_e::cuda>({}, grid),
             table = proxy<execspace_e::cuda>(
                 partition)] __device__(auto bi, auto ci) mutable {
              auto block = grid.block(bi);
              auto mass = block("m", ci);
              if (mass != 0.f) {
                auto vel = block.pack<3>("v", ci);
#if 1
                if ((vel(1) < -5.1 || vel(1) > -4.9) && ci == 0) {
                  auto pos =
                      (table._activeKeys[bi] + grid.cellid_to_coord(ci)) *
                      grid.dx;
                  printf("(%f, %f, %f) mass: %f, vel: %f, %f, %f\n", pos[0],
                         pos[1], pos[2], mass, vel[0], vel[1], vel[2]);
                }
#endif
              }
            });
    puts("done gridupdate check");
    getchar();
#endif

    maxVelSqr->set<float>(velSqr[0]);
    fmt::print(fg(fmt::color::cyan), "done executing GridUpdate\n");
    set_output("ZSGrid", zsgrid);
    set_output("MaxVelSqr", maxVelSqr);
  }
};

ZENDEFNODE(
    UpdateZSGrid,
    {
        {{"float", "gravity", "-9.8"}, "ZSPartition", "ZSGrid", "dt", "Accel"},
        {"ZSGrid", "MaxVelSqr"},
        {},
        {"MPM"},
    });

struct ZSReturnMapping : INode {
  template <typename PM>
  void returnMapping(zs::CudaExecutionPolicy &cudaPol,
                     typename ZenoParticles::particles_t &pars,
                     const zs::StvkWithHencky<float> &elasticModel,
                     const PM &plasticModel) const {
    using namespace zs;
    cudaPol(range(pars.size()),
            [pars = proxy<execspace_e::cuda>({}, pars),
             elasticModel = elasticModel,
             plasticModel = plasticModel] __device__(size_t pi) mutable {
              auto FeHat = pars.pack<3, 3>("F", pi);
              if constexpr (is_same_v<zs::NonAssociativeCamClay<float>,
                                      RM_CVREF_T(plasticModel)>) {
                auto logJp = pars("logJp", pi);
                if (plasticModel.project_strain(FeHat, elasticModel, logJp)) {
                  pars("logJp", pi) = logJp;
                  pars.tuple<9>("F", pi) = FeHat;
                }
              } else { // vm, dp
                if (plasticModel.project_strain(FeHat, elasticModel))
                  pars.tuple<9>("F", pi) = FeHat;
              }
            });
  }
  void return_mapping_surface(zs::CudaExecutionPolicy &cudaPol,
                              typename ZenoParticles::particles_t &eles) const {
    using namespace zs;
    cudaPol(range(eles.size()), [eles = proxy<execspace_e::cuda>(
                                     {}, eles)] __device__(size_t pi) mutable {
#if 1
      auto d = eles.pack<3, 3>("d", pi);
#else
      auto F = eles.pack<3, 3>("F", pi);
#endif
      // hard code ftm
      constexpr auto gamma = 0.f;
      constexpr auto k = 400.f;
      constexpr auto friction_coeff = 0.f;
// constexpr auto friction_coeff = 0.17f;
#if 1
      auto [Q, R] = math::gram_schmidt(d);
#else
      auto [Q, R] = math::gram_schmidt(F);
#endif
      if (gamma == 0.f) {
        R(0, 2) = R(1, 2) = 0;
        R(2, 2) = zs::min(R(2, 2), 1.f);
      } else if (R(2, 2) > 1) {
        R(0, 2) = R(1, 2) = 0;
        R(2, 2) = 1;
      } else if (R(2, 2) <= 0) { // inversion
        R(0, 2) = R(1, 2) = 0;
        R(2, 2) = zs::max(R(2, 2), -1.f);
      } else if (R(2, 2) < 1) {
        auto rr = R(0, 2) * R(0, 2) + R(1, 2) * R(1, 2);
        auto r33_m_1 = R(2, 2) - 1;
#if 0
        auto r11_r22 = R(0, 0) * R(1, 1);
        auto fn = k * r33_m_1 * r33_m_1 / r11_r22; // normal traction
        auto ff = gamma * zs::sqrt(rr) / r11_r22;  // tangential traction
        if (ff >= friction_coeff * fn) {
          auto scale = friction_coeff * fn / ff;
          R(0, 2) *= scale;
          R(1, 2) *= scale;
        }
#else
        auto gamma_over_k = gamma / k;
        auto zz = friction_coeff  * r33_m_1 * r33_m_1; // normal traction
        if (gamma_over_k * gamma_over_k * rr - zz * zz > 0) {
          auto scale = zz / (gamma_over_k * zs::sqrt(rr));
          R(0, 2) *= scale;
          R(1, 2) *= scale;
        }
#endif
      }
#if 1
      d = Q * R;
      eles.tuple<9>("d", pi) = d;
      eles.tuple<9>("F", pi) = d * eles.pack<3, 3>("DmInv", pi);
#else
      F = Q * R;
      eles.tuple<9>("F", pi) = F;
      eles.tuple<9>("d", pi) = F * inverse(eles.pack<3, 3>("DmInv", pi));
#endif
    });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ZSReturnMapping\n");

    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);

    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();
      if (parObjPtr->category == ZenoParticles::mpm) {
        if (parObjPtr->getModel().hasPlasticity()) {
          match(
              [this, &cudaPol,
               &pars](const zs::StvkWithHencky<float> &elasticModel,
                      const auto &plasticModel)
                  -> std::enable_if_t<
                      !is_same_v<RM_CVREF_T(plasticModel), std::monostate>> {
                returnMapping(cudaPol, pars, elasticModel, plasticModel);
              },
              [](...) {
                throw std::runtime_error(
                    "unsupported elasto-plasticity models");
              })(parObjPtr->getModel().getElasticModel(),
                 parObjPtr->getModel().getPlasticModel());
        }
      } else if (parObjPtr->category == ZenoParticles::tracker) {
      } else {
        auto &eles = parObjPtr->getQuadraturePoints();
        if (parObjPtr->category == ZenoParticles::surface)
          return_mapping_surface(cudaPol, eles);
      }
    }

    fmt::print(fg(fmt::color::cyan), "done executing ZSReturnMapping\n");
    set_output("ZSParticles", get_input("ZSParticles"));
  }
};

ZENDEFNODE(ZSReturnMapping, {
                                {"ZSParticles"},
                                {"ZSParticles"},
                                {},
                                {"MPM"},
                            });

} // namespace zeno