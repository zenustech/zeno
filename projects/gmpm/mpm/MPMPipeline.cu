#include "Structures.hpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/simulation/Utils.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

/// sparsity
struct ZSPartitionForZSParticles : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing ZSPartitionForZSParticles\n");
    auto table = get_input<ZenoPartition>("ZSPartition");
    auto &partition = table->get();
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();

    std::vector<ZenoParticles *> parObjPtrs{};
    if (has_input<ZenoParticles>("ZSParticles"))
      parObjPtrs.push_back(get_input<ZenoParticles>("ZSParticles").get());
    else if (has_input<ListObject>("ZSParticles")) {
      auto &objSharedPtrLists = *get_input<ListObject>("ZSParticles");
      for (auto &&objSharedPtr : objSharedPtrLists.get())
        if (auto ptr = dynamic_cast<ZenoParticles *>(objSharedPtr.get());
            ptr != nullptr)
          parObjPtrs.push_back(ptr);
    }

    using namespace zs;
    std::size_t cnt = 0;
    for (auto &&parObjPtr : parObjPtrs)
      cnt += parObjPtr->getParticles().size();
    if (partition._tableSize * 3 / 2 < partition.evaluateTableSize(cnt) ||
        partition._tableSize / 2 < cnt)
      partition.resize(cuda_exec(), cnt);

    using Partition = typename ZenoPartition::table_t;
    auto cudaPol = cuda_exec().device(0);
    cudaPol(range(partition._tableSize),
            [table = proxy<execspace_e::cuda>(partition)] __device__(
                size_t i) mutable {
              table._table.keys[i] =
                  Partition::key_t::uniform(Partition::key_scalar_sentinel_v);
              table._table.indices[i] = Partition::sentinel_v;
              table._table.status[i] = -1;
              if (i == 0)
                *table._cnt = 0;
            });
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
    }
    fmt::print("partition of [{}] blocks for {} particles\n", partition.size(),
               cnt);

    fmt::print(fg(fmt::color::cyan),
               "done executing ZSPartitionForZSParticles\n");
    set_output("ZSPartition", table);
  }
};

ZENDEFNODE(ZSPartitionForZSParticles,
           {
               {"ZSPartition", "ZSGrid", "ZSParticles"},
               {"ZSPartition"},
               {},
               {"MPM"},
           });

struct ExpandZSPartition : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ExpandZSPartition\n");
    auto table = get_input<ZenoPartition>("ZSPartition");
    auto &partition = table->get();
    auto offset = get_param<int>("offset");
    auto extent = get_param<int>("extent");

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);
    using grid_t = typename ZenoGrid::grid_t;
    static_assert(grid_traits<grid_t>::is_power_of_two,
                  "grid side_length should be power of two");

    auto prevCnt = partition.size();
    cudaPol(range(prevCnt), [table = proxy<execspace_e::cuda>(partition),
                             offset, extent] __device__(size_t bi) mutable {
      auto blockid = table._activeKeys[bi];
      for (auto ijk : ndrange<3>(extent))
        table.insert(blockid + (make_vec<int>(ijk) + offset) *
                                   (int)grid_traits<grid_t>::side_length);
    });
    fmt::print("partition insertion [{}] blocks -> [{}] blocks\n", prevCnt,
               partition.size());
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
    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();
    auto cnt = partition.size();

    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();
    grid.resize(cnt);

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);
    // clear grid
    cudaPol({(int)cnt, (int)ZenoGrid::grid_t::block_space()},
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
    auto maxVelSqr = IObject::make<NumericObject>();

    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();
    auto stepDt = get_input<NumericObject>("dt")->get<float>();

    using namespace zs;
    auto gravity = get_param<float>("gravity");
    auto accel = zs::vec<float, 3>::zeros();
    if (has_input("ExtForce")) {
      auto tmp = get_input<NumericObject>("Accel")->get<vec3f>();
      accel = zs::vec<float, 3>{tmp[0], tmp[1], tmp[2]};
    } else
      accel[1] = gravity;

    Vector<float> velSqr{1, zs::memsrc_e::um, 0};
    velSqr[0] = 0;
    auto cudaPol = cuda_exec().device(0);

    cudaPol({(int)partition.size(), (int)ZenoGrid::grid_t::block_space()},
            [grid = proxy<execspace_e::cuda>({}, grid), stepDt, accel,
             ptr = velSqr.data()] __device__(int bi, int ci) mutable {
              auto block = grid.block(bi);
              auto mass = block("m", ci);
              if (mass != 0.f) {
                mass = 1.f / mass;
                auto vel = block.pack<3>("v", ci) * mass;
                vel += accel * stepDt;
                block.set("v", ci, vel);
                /// cfl dt
                auto velSqr = vel.l2NormSqr();
                atomic_max(exec_cuda, ptr, velSqr);
              }
            });

    maxVelSqr->set<float>(velSqr[0]);
    fmt::print(fg(fmt::color::cyan), "done executing GridUpdate\n");
    set_output("ZSGrid", zsgrid);
    set_output("MaxVelSqr", maxVelSqr);
  }
};

ZENDEFNODE(UpdateZSGrid, {
                             {"ZSPartition", "ZSGrid", "dt", "Accel"},
                             {"ZSGrid", "MaxVelSqr"},
                             {{"float", "gravity", "-9.8"}},
                             {"MPM"},
                         });

struct ZSParticleToZSGrid : INode {
  template <typename Model>
  void p2g(zs::CudaExecutionPolicy &cudaPol, const Model &model,
           const float volume, const typename ZenoParticles::particles_t &pars,
           const typename ZenoPartition::table_t &partition, const float dt,
           typename ZenoGrid::grid_t &grid) {
    using namespace zs;
    cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 table = proxy<execspace_e::cuda>(partition),
                                 grid = proxy<execspace_e::cuda>({}, grid), dt,
                                 dxinv = 1.f / grid.dx, vol = volume,
                                 model] __device__(size_t pi) mutable {
      using grid_t = RM_CVREF_T(grid);
      const auto Dinv = 4.f * dxinv * dxinv;
      auto localPos = pars.pack<3>("pos", pi);
      auto vel = pars.pack<3>("vel", pi);
      auto mass = pars("mass", pi);
      auto C = pars.pack<3, 3>("C", pi);
      auto F = pars.pack<3, 3>("F", pi);
      auto P = model.first_piola(F);

      auto contrib = -dt * Dinv * vol * P * F.transpose();
      auto arena = make_local_arena(grid.dx, localPos);

      for (auto loc : arena.range()) {
        auto coord = arena.coord(loc);
        auto localIndex = coord & (grid_t::side_length - 1);
        auto blockno = table.query(coord - localIndex);
        if (blockno < 0)
          printf("THE HELL!");
        auto block = grid.block(blockno);

        auto xixp = arena.diff(loc);
        auto W = arena.weight(loc);
        const auto cellid = grid_t::coord_to_cellid(localIndex);
        atomic_add(exec_cuda, &block("m", cellid), mass * W);
        auto Cxixp = C * xixp;
        auto fdt = contrib * xixp;
        for (int d = 0; d != 3; ++d)
          atomic_add(exec_cuda, &block("v", d, cellid),
                     W * (mass * (vel[d] + Cxixp[d]) + fdt[d]));
      }
    });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ZSParticleToZSGrid\n");

    std::vector<ZenoParticles *> parObjPtrs{};
    if (has_input<ZenoParticles>("ZSParticles"))
      parObjPtrs.push_back(get_input<ZenoParticles>("ZSParticles").get());
    else if (has_input<ListObject>("ZSParticles")) {
      auto &objSharedPtrLists = *get_input<ListObject>("ZSParticles");
      for (auto &&objSharedPtr : objSharedPtrLists.get())
        if (auto ptr = dynamic_cast<ZenoParticles *>(objSharedPtr.get());
            ptr != nullptr)
          parObjPtrs.push_back(ptr);
    }
    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();
    auto stepDt = get_input<zeno::NumericObject>("dt")->get<float>();

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);

    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();
      auto &model = parObjPtr->getModel();

      fmt::print("[p2g] dx: {}, dt: {}, npars: {}\n", grid.dx, stepDt,
                 pars.size());

      match([&](auto &elasticModel) {
        p2g(cudaPol, elasticModel, model.volume, pars, partition, stepDt, grid);
      })(model.getElasticModel());
    }

    fmt::print(fg(fmt::color::cyan), "done executing ZSParticleToZSGrid\n");
    set_output("ZSGrid", zsgrid);
  }
};

ZENDEFNODE(ZSParticleToZSGrid,
           {
               {"ZSParticles", "ZSPartition", "ZSGrid", "dt"},
               {"ZSGrid"},
               {},
               {"MPM"},
           });

struct ZSGridToZSParticle : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ZSGridToZSParticle\n");
    auto &grid = get_input<ZenoGrid>("ZSGrid")->get();
    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();

    std::vector<ZenoParticles *> parObjPtrs{};
    if (has_input<ZenoParticles>("ZSParticles"))
      parObjPtrs.push_back(get_input<ZenoParticles>("ZSParticles").get());
    else if (has_input<ListObject>("ZSParticles")) {
      auto &objSharedPtrLists = *get_input<ListObject>("ZSParticles");
      for (auto &&objSharedPtr : objSharedPtrLists.get())
        if (auto ptr = dynamic_cast<ZenoParticles *>(objSharedPtr.get());
            ptr != nullptr)
          parObjPtrs.push_back(ptr);
    }

    auto stepDt = get_input<NumericObject>("dt")->get<float>();

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);
    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();

      cudaPol(range(pars.size()),
              [pars = proxy<execspace_e::cuda>({}, pars),
               table = proxy<execspace_e::cuda>(partition),
               grid = proxy<execspace_e::cuda>({}, grid), dt = stepDt,
               dxinv = 1.f / grid.dx] __device__(size_t pi) mutable {
                using grid_t = RM_CVREF_T(grid);
                const auto Dinv = 4.f * dxinv * dxinv;
                auto pos = pars.pack<3>("pos", pi);
                auto vel = zs::vec<float, 3>::zeros();
                auto C = zs::vec<float, 3, 3>::zeros();

                auto arena = make_local_arena(grid.dx, pos);
                for (auto loc : arena.range()) {
                  auto coord = arena.coord(loc);
                  auto localIndex = coord & (grid_t::side_length - 1);
                  auto blockno = table.query(coord - localIndex);
                  if (blockno < 0)
                    printf("THE HELL!");
                  auto block = grid.block(blockno);
                  auto xixp = arena.diff(loc);
                  auto W = arena.weight(loc);
                  auto vi =
                      block.pack<3>("v", grid_t::coord_to_cellid(localIndex));

                  vel += vi * W;
                  C += W * Dinv * dyadic_prod(vi, xixp);
                }
                pars.tuple<3>("vel", pi) = vel;
                pars.tuple<3 * 3>("C", pi) = C;
                pos += vel * dt;
                pars.tuple<3>("pos", pi) = pos;

                auto F = pars.pack<3, 3>("F", pi);
                auto tmp = zs::vec<float, 3, 3>::identity() + C * dt;
                F = tmp * F;
                pars.tuple<3 * 3>("F", pi) = F;
              });
    }
    fmt::print(fg(fmt::color::cyan), "done executing ZSGridToZSParticle\n");
  }
};

ZENDEFNODE(ZSGridToZSParticle,
           {
               {"ZSGrid", "ZSPartition", "ZSParticles", "dt"},
               {},
               {},
               {"MPM"},
           });

} // namespace zeno