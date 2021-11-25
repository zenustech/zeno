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

struct MPMStepping : INode {
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

      if (pars.size() < 3000 && pi < 3) {
        auto p = localPos;
        auto v = vel;
        printf("[p2g] pos[%d] (%f, %f, %f) vel (%f, %f, %f) mass (%f)\n",
               (int)pi, p[0], p[1], p[2], v[0], v[1], v[2], mass);
      }

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
    fmt::print(fg(fmt::color::green), "begin executing MPMStepping\n");

    size_t numPars = 0;
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

    for (auto parObjPtr : parObjPtrs)
      numPars += parObjPtr->getParticles().size();

    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();
    auto stepDt = get_input<zeno::NumericObject>("dt")->get<float>();

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);

#if 0
    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();
      cudaPol(range(pars.size()),
              [pars = proxy<execspace_e::cuda>({}, pars),
               table = proxy<execspace_e::cuda>(partition),
               grid = proxy<execspace_e::cuda>({}, grid), dt = stepDt,
               dxinv = 1.f / grid.dx] __device__(size_t pi) mutable {
                using grid_t = RM_CVREF_T(grid);
                auto pos = pars.pack<3>("pos", pi);
                auto vel = pars.pack<3>("vel", pi);
                pos += vel * dt;
                pars.tuple<3>("pos", pi) = pos;
              });
    }
#endif

    /// partition
    using Partition = typename ZenoPartition::table_t;
    using grid_t = typename ZenoGrid::grid_t;
    partition.resize(cudaPol, numPars);
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
    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();
      cudaPol(range(pars.size()),
              [pars = proxy<execspace_e::cuda>({}, pars),
               table = proxy<execspace_e::cuda>(partition),
               dxinv = 1.f / grid.dx] __device__(size_t pi) mutable {
                auto x = pars.pack<3>(1, pi);
                auto c = (x * dxinv - 0.5);
                typename Partition::key_t coord{};
                for (int d = 0; d != 3; ++d)
                  coord[d] = lower_trunc(c[d]);
                table.insert((coord - (coord & (grid_t::side_length - 1))));
              });
    }
    cudaPol(range(partition.size()),
            [table = proxy<execspace_e::cuda>(partition), offset = 0,
             extent = 2] __device__(size_t bi) mutable {
              auto blockid = table._activeKeys[bi];
              for (auto ijk : ndrange<3>(extent))
                table.insert(blockid + (make_vec<int>(ijk) + offset) *
                                           grid_t::side_length);
            });
    /// grid
    grid.resize(1000000);
    // clear grid
    fmt::print("blocksize: {}\n", ZenoGrid::grid_t::block_space());
    cudaPol({(int)partition.size(), (int)ZenoGrid::grid_t::block_space()},
            [grid = proxy<execspace_e::cuda>({}, grid)] __device__(
                int bi, int ci) mutable {
              auto block = grid.block(bi);
              const auto nchns = grid.numChannels();
              for (int i = 0; i != nchns; ++i)
                block(i, ci) = 0;
            });

    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();
      auto &model = parObjPtr->getModel();

      fmt::print("[p2g] dx: {}, dt: {}, npars: {}\n", grid.dx, stepDt,
                 pars.size());

      match([&](auto &elasticModel) {
        p2g(cudaPol, elasticModel, model.volume, pars, partition, stepDt, grid);
      })(model.getElasticModel());
    }

    {
      /// particle momentum
      Vector<float> lm{3, memsrc_e::um, 0}, am{3, memsrc_e::um, 0};
      for (int d = 0; d != 3; ++d)
        lm[d] = am[d] = 0;
      auto &pars = parObjPtrs[0]->getParticles();
      cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                   lm = proxy<execspace_e::cuda>(lm),
                                   am = proxy<execspace_e::cuda>(am),
                                   dx = grid.dx] __device__(size_t pi) {
        auto mass = pars("mass", pi);
        auto pos = pars.pack<3>("pos", pi);
        auto vel = pars.pack<3>("vel", pi);
        const auto Dinv = 4.f / dx / dx;
        auto B = pars.pack<3, 3>("C", pi) / Dinv;
        for (int d = 0; d != 3; ++d)
          atomic_add(exec_cuda, &lm[d], mass * vel[d]);

        auto res = pos.cross(vel) * mass;
        for (int a = 0; a != 3; ++a)
          for (int b = 0; b != 3; ++b)
            for (int g = 0; g != 3; ++g)
              if ((a == 0 && b == 1 && g == 2) ||
                  (a == 1 && b == 2 && g == 0) || (a == 2 && b == 0 && g == 1))
                res(g) += mass * B(a, b);
              else if ((a == 0 && b == 2 && g == 1) ||
                       (a == 1 && b == 0 && g == 2) ||
                       (a == 2 && b == 1 && g == 0))
                res(g) -= mass * B(a, b);

        for (int d = 0; d != 3; ++d)
          atomic_add(exec_cuda, &am[d], res[d]);
      });
      fmt::print("[[after p2g]] pars({} total) linear momentum: ({}, {}, {}), "
                 "angular momentum: ({}, {}, {})\n",
                 pars.size(), lm[0], lm[1], lm[2], am[0], am[1], am[2]);
    }

    auto cnt = partition.size();
    cudaPol({(int)partition.size(), (int)ZenoGrid::grid_t::block_space()},
            [grid = proxy<execspace_e::cuda>({}, grid),
             stepDt] __device__(int bi, int ci) mutable {
              auto block = grid.block(bi);
              auto mass = block("m", ci);
              if (mass != 0.f) {
                mass = 1.f / mass;
                auto vel = block.pack<3>("v", ci) * mass;
                // vel += extf * stepDt;
                // vel[1] += -9.8 * stepDt;
                /// write back
                block.set("v", ci, vel);
              }
            });
    fmt::print("grid update numblocks {}\n", cnt);

    {
      Vector<float> lm{3, memsrc_e::um, 0}, am{3, memsrc_e::um, 0};
      /// particle momentum
      for (int d = 0; d != 3; ++d)
        lm[d] = am[d] = 0;
      cudaPol({(int)partition.size(), (int)ZenoGrid::grid_t::block_space()},
              [grid = proxy<execspace_e::cuda>({}, grid),
               table = proxy<execspace_e::cuda>(partition),
               lm = proxy<execspace_e::cuda>(lm),
               am = proxy<execspace_e::cuda>(am)] __device__(int bi, int ci) {
                using grid_t = RM_CVREF_T(grid);
                auto block = grid.block(bi);
                auto mass = block("m", ci);
                if (mass != 0) {
                  auto blockkey = table._activeKeys[bi];
                  auto x = (blockkey + grid_t::cellid_to_coord(ci)) * grid.dx;
                  auto mv = block.pack<3>("v", ci) * mass;
                  /// x cross mv;
                  auto res = x.cross(mv);

                  for (int i = 0; i != 3; ++i) {
                    atomic_add(exec_cuda, &am[i], res[i]);
                    atomic_add(exec_cuda, &lm[i], mv[i]);
                  }
                }
              });
      fmt::print("[[after grid update]] grid({} blocks total) linear momentum: "
                 "({}, {}, {}), angular "
                 "momentum: ({}, {}, {})\n",
                 partition.size(), lm[0], lm[1], lm[2], am[0], am[1], am[2]);
    }

    Vector<double> vols{2, memsrc_e::um, 0};
    vols[0] = vols[1] = 0.;
    cudaPol({(int)partition.size(), (int)ZenoGrid::grid_t::block_space()},
            [grid = proxy<execspace_e::cuda>({}, grid),
             vols = proxy<execspace_e::cuda>(vols),
             stepDt] __device__(int bi, int ci) mutable {
              auto block = grid.block(bi);
              auto mass = block("m", ci);
              if (mass != 0.f) {
                atomic_add(exec_cuda, &vols[0],
                           (double)(grid.dx * grid.dx * grid.dx));
                atomic_add(exec_cuda, &vols[1], 1.);
              }
            });

    fmt::print(fg(fmt::color::red),
               "sum total particle volume: {} x {} => {}\n", numPars,
               parObjPtrs[0]->getModel().volume,
               parObjPtrs[0]->getModel().volume * numPars);
    fmt::print(fg(fmt::color::brown), "sum total grid volume: {} x {} => {}\n",
               vols[1], grid.dx * grid.dx * grid.dx, vols[0]);

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

    fmt::print(fg(fmt::color::cyan), "done executing MPMStepping\n");
  }
};

ZENDEFNODE(MPMStepping, {
                            {"ZSParticles", "ZSPartition", "ZSGrid", "dt"},
                            {},
                            {},
                            {"MPM"},
                        });

} // namespace zeno