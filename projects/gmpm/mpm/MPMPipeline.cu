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
    if (cached && table->hasTags()) {
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
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ExpandZSPartition\n");
    auto table = get_input<ZenoPartition>("ZSPartition");
    auto &partition = table->get();
    auto offset = get_param<int>("offset");
    auto extent = get_param<int>("extent");

    if (!table->rebuilt) { // only expand after a fresh rebuilt
      fmt::print(fg(fmt::color::cyan), "done executing ExpandZSPartition "
                                       "(skipping expansion due to caching)\n");
      set_output("ZSPartition", std::move(table));
      return;
    }

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
    if (table->hasTags())
      identify_boundary_indices(cudaPol, *table, wrapv<grid_t::side_length>{});

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

    if (zsgrid->transferScheme == "apic")
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
    else if (zsgrid->transferScheme == "flip")
      cudaPol(Collapse{partition.size(), ZenoGrid::grid_t::block_space()},
              [grid = proxy<execspace_e::cuda>({}, grid), stepDt, accel,
               ptr = velSqr.data()] __device__(int bi, int ci) mutable {
                auto block = grid.block(bi);
                auto mass = block("m", ci);
                if (mass != 0.f) {
                  mass = 1.f / mass;

                  auto vel = block.pack<3>("v", ci) * mass;
                  vel += accel * stepDt;
                  block.set("v", ci, vel);

                  auto vdiff = block.pack<3>("vdiff", ci) * mass;
                  vdiff += accel * stepDt;
                  block.set("vdiff", ci, vdiff);

                  /// cfl dt
                  auto velSqr = vel.l2NormSqr();
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
                  auto nrm = block.pack<3>("nrm", ci);
                  if (auto len = nrm.l2NormSqr();
                      len > zs::limits<float>::epsilon() * 128)
                    nrm /= zs::sqrt(len);
                  block.set("nrm", ci, nrm);
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

struct ApplyGridBoundaryOnZSGrid : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing ApplyGridBoundaryOnZSGrid\n");

    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();
    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();
    auto bg = get_input<ZenoGrid>("BoundaryZSGrid");
    if (bg->transferScheme != "boundary")
      throw std::runtime_error("boundary grid is not of boundary type!");
    auto &boundaryGrid = get_input<ZenoGrid>("BoundaryZSGrid")->get();
    auto &boundaryPartition =
        get_input<ZenoPartition>("BoundaryZSPartition")->get();

    using namespace zs;

    auto cudaPol = cuda_exec().device(0);
    auto typeStr = get_param<std::string>("type");
    collider_e type =
        typeStr == "sticky"
            ? collider_e::Sticky
            : (typeStr == "slip" ? collider_e::Slip : collider_e::Separate);

    cudaPol(Collapse{boundaryPartition.size(), boundaryGrid.block_size},
            [grid = proxy<execspace_e::cuda>({}, grid),
             boundaryGrid = proxy<execspace_e::cuda>({}, boundaryGrid),
             table = proxy<execspace_e::cuda>(partition),
             boundaryTable = proxy<execspace_e::cuda>(boundaryPartition),
             type] __device__(int bi, int ci) mutable {
              using table_t = RM_CVREF_T(table);
              auto boundaryBlock = boundaryGrid.block(bi);
              if (boundaryBlock("m", ci) == 0.f)
                return;
              auto blockid = boundaryTable._activeKeys[bi];
              auto blockno = table.query(blockid);
              if (blockno == table_t::sentinel_v)
                return;

              auto block = grid.block(blockno);
              if (block("m", ci) == 0.f)
                return;
              if (type == collider_e::Sticky)
                block.set("v", ci, boundaryBlock.pack<3>("v", ci));
              else {
                auto v_object = boundaryBlock.pack<3>("v", ci);
                auto normal = boundaryBlock.pack<3>("nrm", ci);
                auto v = block.pack<3>("v", ci);
                v -= v_object;
                auto proj = normal.dot(v);
                if ((type == collider_e::Separate && proj < 0) ||
                    type == collider_e::Slip)
                  v -= proj * normal;
                v += v_object;
                block.set("v", ci, v);
              }
            });

    fmt::print(fg(fmt::color::cyan),
               "done executing ApplyGridBoundaryOnZSGrid\n");
    set_output("ZSGrid", zsgrid);
  }
};

ZENDEFNODE(ApplyGridBoundaryOnZSGrid,
           {
               {"ZSPartition", "ZSGrid", "BoundaryZSPartition",
                "BoundaryZSGrid"},
               {"ZSGrid"},
               {{"enum sticky slip separate", "type", "sticky"}},
               {"MPM"},
           });

struct ApplyBoundaryOnZSGrid : INode {
  template <typename LsView>
  constexpr void
  projectBoundary(zs::CudaExecutionPolicy &cudaPol, LsView lsv,
                  const ZenoBoundary &boundary,
                  const typename ZenoPartition::table_t &partition,
                  typename ZenoGrid::grid_t &grid,
                  std::string_view transferScheme) {
    using namespace zs;
    auto collider = boundary.getBoundary(lsv);
    if (transferScheme == "apic")
      cudaPol(Collapse{partition.size(), ZenoGrid::grid_t::block_space()},
              [grid = proxy<execspace_e::cuda>({}, grid),
               table = proxy<execspace_e::cuda>(partition),
               boundary = collider] __device__(int bi, int ci) mutable {
                auto block = grid.block(bi);
                auto mass = block("m", ci);
                if (mass != 0.f) {
                  auto vel = block.pack<3>("v", ci);
                  auto pos =
                      (table._activeKeys[bi] + grid.cellid_to_coord(ci)) *
                      grid.dx;
                  boundary.resolveCollision(pos, vel);
                  block.set("v", ci, vel);
                }
              });
    else if (transferScheme == "flip")
      cudaPol(Collapse{partition.size(), ZenoGrid::grid_t::block_space()},
              [grid = proxy<execspace_e::cuda>({}, grid),
               table = proxy<execspace_e::cuda>(partition),
               boundary = collider] __device__(int bi, int ci) mutable {
                auto block = grid.block(bi);
                auto mass = block("m", ci);
                if (mass != 0.f) {
                  auto vel = block.pack<3>("v", ci);
                  auto pos =
                      (table._activeKeys[bi] + grid.cellid_to_coord(ci)) *
                      grid.dx;
                  boundary.resolveCollision(pos, vel);
                  block.set("v", ci, vel);

                  auto vdiff = block.pack<3>("vdiff", ci);
                  boundary.resolveCollision(pos, vdiff);
                  block.set("vdiff", ci, vdiff);
                }
              });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing ApplyBoundaryOnZSGrid\n");

    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();

    using namespace zs;

    auto cudaPol = cuda_exec().device(0);

    if (has_input<ZenoBoundary>("ZSBoundary")) {
      auto boundary = get_input<ZenoBoundary>("ZSBoundary");
      auto &partition = get_input<ZenoPartition>("ZSPartition")->get();

      using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
      using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
      using const_transition_ls_t =
          typename ZenoLevelSet::const_transition_ls_t;
      if (boundary->zsls)
        match([&](const auto &ls) {
          if constexpr (is_same_v<RM_CVREF_T(ls), basic_ls_t>) {
            match([&](const auto &lsPtr) {
              auto lsv = get_level_set_view<execspace_e::cuda>(lsPtr);
              projectBoundary(cudaPol, lsv, *boundary, partition, grid,
                              zsgrid->transferScheme);
            })(ls._ls);
          } else if constexpr (is_same_v<RM_CVREF_T(ls), const_sdf_vel_ls_t>) {
            match([&](auto lsv) {
              projectBoundary(cudaPol, SdfVelFieldView{lsv}, *boundary,
                              partition, grid, zsgrid->transferScheme);
            })(ls.template getView<execspace_e::cuda>());
          } else if constexpr (is_same_v<RM_CVREF_T(ls),
                                         const_transition_ls_t>) {
            match([&](auto fieldPair) {
              auto &fvSrc = std::get<0>(fieldPair);
              auto &fvDst = std::get<1>(fieldPair);
              projectBoundary(cudaPol,
                              TransitionLevelSetView{SdfVelFieldView{fvSrc},
                                                     SdfVelFieldView{fvDst},
                                                     ls._stepDt, ls._alpha},
                              *boundary, partition, grid,
                              zsgrid->transferScheme);
            })(ls.template getView<zs::execspace_e::cuda>());
          }
        })(boundary->zsls->getLevelSet());
    }

    fmt::print(fg(fmt::color::cyan), "done executing ApplyBoundaryOnZSGrid \n");
    set_output("ZSGrid", zsgrid);
  }
};

ZENDEFNODE(ApplyBoundaryOnZSGrid, {
                                      {"ZSPartition", "ZSGrid", "ZSBoundary"},
                                      {"ZSGrid"},
                                      {},
                                      {"MPM"},
                                  });

struct ZSParticleToZSGrid : INode {
  void p2g_apic_momentum(zs::CudaExecutionPolicy &cudaPol,
                         const typename ZenoParticles::particles_t &pars,
                         const typename ZenoPartition::table_t &partition,
                         typename ZenoGrid::grid_t &grid) {
    using namespace zs;
    cudaPol(range(pars.size()),
            [pars = proxy<execspace_e::cuda>({}, pars),
             table = proxy<execspace_e::cuda>(partition),
             grid = proxy<execspace_e::cuda>({}, grid),
             dxinv = 1.f / grid.dx] __device__(size_t pi) mutable {
              using grid_t = RM_CVREF_T(grid);
              const auto Dinv = 4.f * dxinv * dxinv;
              auto pos = pars.pack<3>("pos", pi);
              auto vel = pars.pack<3>("vel", pi);
              auto mass = pars("mass", pi);
              auto C = pars.pack<3, 3>("C", pi);

#if 0
              if (pi < 10) {
                printf("(%f, %f, %f) vel: %f, %f, %f, mass: %f, vol: %f\n",
                       pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], mass,
                       vol);
              }
#endif

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
                const auto cellid = grid_t::coord_to_cellid(localIndex);
                atomic_add(exec_cuda, &block("m", cellid), mass * W);
                auto Cxixp = C * xixp;
                for (int d = 0; d != 3; ++d)
                  atomic_add(exec_cuda, &block("v", d, cellid),
                             W * mass * (vel[d] + Cxixp[d]));
              }
            });

#if 0
    cudaPol(Collapse{pars.size()}, [pars = proxy<execspace_e::cuda>({}, pars),
                                    table = proxy<execspace_e::cuda>(partition),
                                    grid = proxy<execspace_e::cuda>({}, grid),
                                    dxinv =
                                        1.f /
                                        grid.dx] __device__(auto pi) mutable {
      using grid_t = RM_CVREF_T(grid);
      auto pos = pars.pack<3>("pos", pi);

      auto c = (pos * dxinv - 0.5);
      typename ZenoPartition::table_t::key_t coord{};
      for (int d = 0; d != 3; ++d)
        coord[d] = lower_trunc(c[d]);
      auto bno = table.query(coord - (coord & (grid_t::side_length - 1)));
      auto cno = grid_t::coord_to_cellid((coord & (grid_t::side_length - 1)));

      auto vel = pars.pack<3>("vel", pi);
      if ((vel(1) < -5.1 || vel(1) > -4.9)) {
        printf(
            "par[%d], [bi, ci]: (%d, %d). pos: %f, %f, %f; vel: %f, %f, %f\n",
            (int)pi, (int)bno, (int)cno, pos[0], pos[1], pos[2], vel[0], vel[1],
            vel[2]);
      }
    });
    puts("done p2g_momentum check");
    getchar();
#endif
  }
  template <typename Model>
  void p2g_surface_force(zs::CudaExecutionPolicy &cudaPol, const Model &model,
                         const typename ZenoParticles::particles_t &verts,
                         const typename ZenoParticles::particles_t &eles,
                         const typename ZenoPartition::table_t &partition,
                         const float dt, typename ZenoGrid::grid_t &grid) {
    using namespace zs;
    cudaPol(range(eles.size()),
            [verts = proxy<execspace_e::cuda>({}, verts),
             eles = proxy<execspace_e::cuda>({}, eles),
             table = proxy<execspace_e::cuda>(partition),
             grid = proxy<execspace_e::cuda>({}, grid), model, dt,
             dxinv = 1.f / grid.dx] __device__(size_t pi) mutable {
              using grid_t = RM_CVREF_T(grid);
              const auto Dinv = 4.f * dxinv * dxinv;
              auto pos = eles.pack<3>("pos", pi);
              auto vel = eles.pack<3>("vel", pi);
              auto mass = eles("mass", pi);
              auto vol = eles("vol", pi);
              auto C = eles.pack<3, 3>("C", pi);
              auto F = eles.pack<3, 3>("F", pi);
              auto d_ = eles.pack<3, 3>("d", pi);

              // hard coded P compute
              using mat2 = zs::vec<float, 2, 2>;
              using mat3 = zs::vec<float, 3, 3>;
              constexpr auto gamma = 0.f;
              constexpr auto k = 100.f;
              auto [Q, R] = math::gram_schmidt(F);
              mat2 R2{R(0, 0), R(0, 1), R(1, 0), R(1, 1)};
              auto P2 = model.first_piola(R2); // use as F
              auto Pplane = mat3::zeros();
              Pplane(0, 0) = P2(0, 0);
              Pplane(0, 1) = P2(0, 1);
              Pplane(1, 0) = P2(1, 0);
              Pplane(1, 1) = P2(1, 1);
              Pplane = Q * Pplane; // inplane

              float rr = R(0, 2) * R(0, 2) + R(1, 2) * R(1, 2);
              float gg = gamma; // normal shearing

              float gf = 0.f;
              if (R(2, 2) < 1) { // compression
                const auto v = 1.f - R(2, 2);
                gf = -k * v * v;
              }

              auto A = mat3::zeros();
              A(0, 0) = gg * R(0, 2) * R(0, 2);
              A(0, 1) = gg * R(0, 2) * R(1, 2);
              A(0, 2) = gg * R(0, 2) * R(2, 2);
              A(1, 1) = gg * R(1, 2) * R(1, 2);
              A(1, 2) = gg * R(1, 2) * R(2, 2);
              A(2, 2) = gf * R(2, 2);
              A(1, 0) = A(0, 1);
              A(2, 0) = A(0, 2);
              A(2, 1) = A(1, 2);
              auto P = Pplane + Q * A * inverse(R).transpose();

              //
              auto P_c3 = col(P, 2);
              auto d_c3 = col(d_, 2);

              auto arena =
                  make_local_arena<grid_e::collocated, kernel_e::quadratic, 1>(
                      grid.dx, pos);
              // compression
              for (auto loc : arena.range()) {
                auto coord = arena.coord(loc);
                auto localIndex = coord & (grid_t::side_length - 1);
                auto blockno = table.query(coord - localIndex);
                if (blockno < 0)
                  printf("THE HELL!");
                auto block = grid.block(blockno);

                auto Wgrad = arena.weightGradients(loc) * dxinv;
                const auto cellid = grid_t::coord_to_cellid(localIndex);

                auto vft = P_c3 * Wgrad.dot(d_c3) * (-vol * dt);
                for (int d = 0; d != 3; ++d)
                  atomic_add(exec_cuda, &block("v", d, cellid), vft(d));
              }

              // type (ii)
              auto transfer = [&P, &grid, &table](const auto &pos,
                                                  const auto &Dinv_r,
                                                  const auto coeff) {
                auto vft =
                    coeff * zs::vec<float, 3>{
                                P(0, 0) * Dinv_r(0) + P(0, 1) * Dinv_r(1),
                                P(1, 0) * Dinv_r(0) + P(1, 1) * Dinv_r(1),
                                P(2, 0) * Dinv_r(0) + P(2, 1) * Dinv_r(1)};
                auto arena = make_local_arena(grid.dx, pos);

                for (auto loc : arena.range()) {
                  auto coord = arena.coord(loc);
                  auto localIndex = coord & (grid_t::side_length - 1);
                  auto blockno = table.query(coord - localIndex);
                  if (blockno < 0)
                    printf("THE HELL!");
                  auto block = grid.block(blockno);

                  auto W = arena.weight(loc);
                  const auto cellid = grid_t::coord_to_cellid(localIndex);
                  for (int d = 0; d != 3; ++d)
                    atomic_add(exec_cuda, &block("v", d, cellid), W * vft[d]);
                }
              };
              auto Dminv = eles.pack<3, 3>("DmInv", pi);
              auto ind0 = (int)eles("inds", (int)0, pi);
              // auto vol0 = verts("vol", ind0);
              auto p0 = verts.pack<3>("pos", ind0);
              {
                zs::vec<float, 3> Dminv_r[2] = {row(Dminv, 0), row(Dminv, 1)};
                transfer(p0, Dminv_r[0] + Dminv_r[1], vol * dt);
                for (int i = 1; i != 3; ++i) {
                  // auto Dinv_ri = row(Dminv, i - 1);
                  auto Dinv_ri = Dminv_r[i - 1];
                  auto ind = (int)eles("inds", (int)i, pi);
                  auto p_i = verts.pack<3>("pos", ind);
                  transfer(p_i, Dinv_ri, -vol * dt);
                  // transfer(p0, Dinv_ri, vol * dt);
                }
              }
            });
  }
  template <typename Model, typename AnisoModel>
  void p2g_apic(zs::CudaExecutionPolicy &cudaPol, const Model &model,
                const AnisoModel &anisoModel,
                const typename ZenoParticles::particles_t &pars,
                const typename ZenoPartition::table_t &partition,
                const float dt, typename ZenoGrid::grid_t &grid) {
    using namespace zs;
    bool materialParamOverride =
        pars.hasProperty("mu") && pars.hasProperty("lam");
    cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 table = proxy<execspace_e::cuda>(partition),
                                 grid = proxy<execspace_e::cuda>({}, grid), dt,
                                 dxinv = 1.f / grid.dx, model = model,
                                 materialParamOverride,
                                 anisoModel] __device__(size_t pi) mutable {
      using grid_t = RM_CVREF_T(grid);
      const auto Dinv = 4.f * dxinv * dxinv;
      auto localPos = pars.pack<3>("pos", pi);
      auto vel = pars.pack<3>("vel", pi);
      auto mass = pars("mass", pi);
      auto vol = pars("vol", pi);
      auto C = pars.pack<3, 3>("C", pi);
      auto F = pars.pack<3, 3>("F", pi);
      if (materialParamOverride) {
        model.mu = pars("mu", pi);
        model.lam = pars("lam", pi);
      }
      auto P = model.first_piola(F);
      if constexpr (is_same_v<RM_CVREF_T(anisoModel), AnisotropicArap<float>>)
        P += anisoModel.first_piola(F, pars.pack<3>("a", pi));

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
  template <typename Model, typename AnisoModel>
  void p2g_flip(zs::CudaExecutionPolicy &cudaPol, const Model &model,
                const AnisoModel &anisoModel,
                const typename ZenoParticles::particles_t &pars,
                const typename ZenoPartition::table_t &partition,
                const float dt, typename ZenoGrid::grid_t &grid) {
    using namespace zs;
    cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 table = proxy<execspace_e::cuda>(partition),
                                 grid = proxy<execspace_e::cuda>({}, grid), dt,
                                 dxinv = 1.f / grid.dx, model,
                                 anisoModel] __device__(size_t pi) mutable {
      using grid_t = RM_CVREF_T(grid);
      auto localPos = pars.pack<3>("pos", pi);
      auto vel = pars.pack<3>("vel", pi);
      auto mass = pars("mass", pi);
      auto vol = pars("vol", pi);
      auto F = pars.pack<3, 3>("F", pi);
      auto P = model.first_piola(F);
      if constexpr (is_same_v<RM_CVREF_T(anisoModel), AnisotropicArap<float>>)
        P += anisoModel.first_piola(F, pars.pack<3>("a", pi));

      auto contrib = -dt * vol * P * F.transpose();
      auto arena = make_local_arena<grid_e::collocated, kernel_e::quadratic, 1>(
          grid.dx, localPos);

      for (auto loc : arena.range()) {
        auto coord = arena.coord(loc);
        auto localIndex = coord & (grid_t::side_length - 1);
        auto blockno = table.query(coord - localIndex);
        if (blockno < 0)
          printf("THE HELL!");
        auto block = grid.block(blockno);

        auto massW = arena.weight(loc) * mass;
        auto Wgrad = arena.weightGradients(loc) * dxinv;
        const auto cellid = grid_t::coord_to_cellid(localIndex);

        atomic_add(exec_cuda, &block("m", cellid), massW);
        auto fdt = contrib * Wgrad;
        for (int d = 0; d != 3; ++d) {
          atomic_add(exec_cuda, &block("v", d, cellid),
                     massW * vel[d] + fdt[d]);
          atomic_add(exec_cuda, &block("vdiff", d, cellid), fdt[d]);
        }
      }
    });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ZSParticleToZSGrid\n");

    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();
    auto stepDt = get_input<zeno::NumericObject>("dt")->get<float>();

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);

#if 0
    cudaPol(Collapse{partition.size(), ZenoGrid::grid_t::block_space()},
            [grid = proxy<execspace_e::cuda>({}, grid),
             table = proxy<execspace_e::cuda>(
                 partition)] __device__(auto bi, auto ci) mutable {
              auto block = grid.block(bi);
              auto mass = block("m", ci);
                auto vel = block.pack<3>("v", ci);
                if (mass != 0.f || vel(1) != 0.f) {
                  auto pos =
                      (table._activeKeys[bi] + grid.cellid_to_coord(ci)) *
                      grid.dx;
                  printf("(%f, %f, %f) mass: %f, vel: %f, %f, %f\n", pos[0],
                         pos[1], pos[2], mass, vel[0], vel[1], vel[2]);
                }
            });
    puts("before p2g grid check");
    getchar();
#endif

    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();
      auto &model = parObjPtr->getModel();

      fmt::print("[p2g] dx: {}, dt: {}, npars: {}\n", grid.dx, stepDt,
                 pars.size());

      if (parObjPtr->category == ZenoParticles::mpm)
        match([&](auto &elasticModel, auto &anisoElasticModel) {
          if (zsgrid->transferScheme == "apic")
            p2g_apic(cudaPol, elasticModel, anisoElasticModel, pars, partition,
                     stepDt, grid);
          else if (zsgrid->transferScheme == "flip")
            p2g_flip(cudaPol, elasticModel, anisoElasticModel, pars, partition,
                     stepDt, grid);
        })(model.getElasticModel(), model.getAnisoElasticModel());
      else if (parObjPtr->category == ZenoParticles::surface) {
        auto &eles = parObjPtr->getQuadraturePoints();
        p2g_apic_momentum(cudaPol, pars, partition, grid);
        p2g_apic_momentum(cudaPol, eles, partition, grid);
        match([&](auto &elasticModel) {
          if (parObjPtr->category == ZenoParticles::surface) {
            p2g_surface_force(cudaPol, elasticModel, pars, eles, partition,
                              stepDt, grid);
          }
        })(model.getElasticModel());
      } else if (parObjPtr->category != ZenoParticles::tracker) {
        // not implemented yet
      }
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
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();
    auto table = get_input<ZenoPartition>("ZSPartition");
    auto &partition = table->get();

    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");

    auto stepDt = get_input<NumericObject>("dt")->get<float>();

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);

    using grid_t = RM_CVREF_T(grid);
    if (table->hasTags()) {
      using Ti = typename ZenoPartition::Ti;
      table->clearTags();
      cudaPol(Collapse{table->numBoundaryEntries(), grid_t::block_size},
              [table = proxy<execspace_e::cuda>(partition),
               boundaryIndices =
                   proxy<execspace_e::cuda>(table->getBoundaryIndices()),
               tags = proxy<execspace_e::cuda>(table->getTags()),
               grid = proxy<execspace_e::cuda>(
                   {}, grid)] __device__(auto boundaryNo, auto ci) mutable {
                using grid_t = RM_CVREF_T(grid);
                using table_t = RM_CVREF_T(table);
                using key_t = typename table_t::key_t;
                auto bi = boundaryIndices[boundaryNo];
                if (grid("m", bi, ci) != 0.f && tags[boundaryNo] == 0) {
                  if (atomic_cas(exec_cuda, &tags[boundaryNo], 0, 1) == 0) {
#if 0
                    auto bcoord = table._activeKeys[bi];
                    constexpr auto side_length = grid_t::side_length;
                    int isBoundary =
                        (table.query(bcoord + key_t{-side_length, 0, 0}) ==
                         table_t::sentinel_v)
                            << 0 |
                        (table.query(bcoord + key_t{side_length, 0, 0}) ==
                         table_t::sentinel_v)
                            << 1 |
                        (table.query(bcoord + key_t{0, -side_length, 0}) ==
                         table_t::sentinel_v)
                            << 2 |
                        (table.query(bcoord + key_t{0, side_length, 0}) ==
                         table_t::sentinel_v)
                            << 3 |
                        (table.query(bcoord + key_t{0, 0, -side_length}) ==
                         table_t::sentinel_v)
                            << 4 |
                        (table.query(bcoord + key_t{0, 0, side_length}) ==
                         table_t::sentinel_v)
                            << 5;
                    printf("grid (%d, %d) [bou tag: %d] mass: %f\n", bi, ci,
                           isBoundary, grid("m", bi, ci));
#endif
                  }
                }
              });
      fmt::print("checking {} boundary blocks out of {} blocks in total.",
                 table->numBoundaryEntries(), table->numEntries());
    }

    for (auto &&parObjPtr : parObjPtrs) {
      if (parObjPtr->asBoundary)
        continue;
      fmt::print("g2p iterating par: {}\n", (void *)parObjPtr);
      auto &pars = parObjPtr->getParticles();

      if (parObjPtr->category == ZenoParticles::mpm) {
        if (zsgrid->transferScheme == "apic")
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
                      auto vi = block.pack<3>(
                          "v", grid_t::coord_to_cellid(localIndex));

                      vel += vi * W;
                      C += W * Dinv * dyadic_prod(vi, xixp);
                    }
                    pars.tuple<3>("vel", pi) = vel;
#if 0
                    // temporal measure for explicit timestepping stability
                    auto skew = 0.5f * (C - C.transpose());
                    auto sym = 0.5f * (C + C.transpose());
                    C = skew + sym * 0.8;
#endif
                    pars.tuple<3 * 3>("C", pi) = C;
                    pos += vel * dt;
                    pars.tuple<3>("pos", pi) = pos;

                    auto F = pars.pack<3, 3>("F", pi);
                    auto tmp = zs::vec<float, 3, 3>::identity() + C * dt;
                    F = tmp * F;
                    pars.tuple<3 * 3>("F", pi) = F;
                  });
        else if (zsgrid->transferScheme == "flip")
          cudaPol(range(pars.size()),
                  [pars = proxy<execspace_e::cuda>({}, pars),
                   table = proxy<execspace_e::cuda>(partition),
                   grid = proxy<execspace_e::cuda>({}, grid), dt = stepDt,
                   dxinv = 1.f / grid.dx] __device__(size_t pi) mutable {
                    using grid_t = RM_CVREF_T(grid);
                    auto pos = pars.pack<3>("pos", pi);
                    auto v = zs::vec<float, 3>::zeros();
                    auto vdiff = zs::vec<float, 3>::zeros();
                    auto vGrad = zs::vec<float, 3, 3>::zeros();

                    auto arena =
                        make_local_arena<grid_e::collocated,
                                         kernel_e::quadratic, 1>(grid.dx, pos);
                    for (auto loc : arena.range()) {
                      auto coord = arena.coord(loc);
                      auto localIndex = coord & (grid_t::side_length - 1);
                      auto blockno = table.query(coord - localIndex);
                      if (blockno < 0)
                        printf("THE HELL!");
                      auto block = grid.block(blockno);
                      auto W = arena.weight(loc);
                      auto Wgrad = arena.weightGradients(loc) * dxinv;

                      auto vi = block.pack<3>(
                          "v", grid_t::coord_to_cellid(localIndex));
                      auto vd = block.pack<3>(
                          "vdiff", grid_t::coord_to_cellid(localIndex));
                      v += vi * W;
                      vdiff += vd * W;
                      vGrad += dyadic_prod(vi, Wgrad);
                    }
                    constexpr float flip = 0.99f;
                    auto vp0 = pars.pack<3>("vel", pi);
                    auto vel = v * (1.f - flip) + (vdiff + vp0) * flip;
                    pars.tuple<3>("vel", pi) = vel;
                    // pos += v * dt; // flip!
                    pos += vel * dt; // asflip!
                    pars.tuple<3>("pos", pi) = pos;

                    auto F = pars.pack<3, 3>("F", pi);
                    auto tmp = zs::vec<float, 3, 3>::identity() + vGrad * dt;
                    F = tmp * F;
                    pars.tuple<3 * 3>("F", pi) = F;
                  });
      } else if (parObjPtr->category == ZenoParticles::tracker) {
        cudaPol(range(pars.size()),
                [pars = proxy<execspace_e::cuda>({}, pars),
                 table = proxy<execspace_e::cuda>(partition),
                 grid = proxy<execspace_e::cuda>({}, grid), dt = stepDt,
                 dxinv = 1.f / grid.dx] __device__(size_t pi) mutable {
                  using grid_t = RM_CVREF_T(grid);
                  auto pos = pars.pack<3>("pos", pi);
                  auto vel = zs::vec<float, 3>::zeros();

                  auto arena = make_local_arena(grid.dx, pos);
                  for (auto loc : arena.range()) {
                    auto coord = arena.coord(loc);
                    auto localIndex = coord & (grid_t::side_length - 1);
                    auto blockno = table.query(coord - localIndex);
                    if (blockno < 0)
                      printf("THE HELL!");
                    auto block = grid.block(blockno);
                    auto W = arena.weight(loc);
                    auto vi =
                        block.pack<3>("v", grid_t::coord_to_cellid(localIndex));

                    vel += vi * W;
                  }
                  // vel
                  pars.tuple<3>("vel", pi) = vel;
                  // pos
                  pos += vel * dt;
                  pars.tuple<3>("pos", pi) = pos;
                });
      } else if (parObjPtr->category != ZenoParticles::mpm) {
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
                    // auto Wgrad = arena.weightGradients(loc) * dxinv;
                    auto vi =
                        block.pack<3>("v", grid_t::coord_to_cellid(localIndex));

                    vel += vi * W;
                    C += W * Dinv * dyadic_prod(vi, xixp);
                  }
                  // vel
                  pars.tuple<3>("vel", pi) = vel;
                  // C
                  auto skew = 0.5f * (C - C.transpose());
                  auto sym = 0.5f * (C + C.transpose());
                  C = skew + sym;
                  pars.tuple<9>("C", pi) = C;
                  // pos
                  pos += vel * dt;
                  pars.tuple<3>("pos", pi) = pos;
                });

#if 0
        cudaPol(Collapse{pars.size()},
                [pars = proxy<execspace_e::cuda>({}, pars),
                 table = proxy<execspace_e::cuda>(partition),
                 grid = proxy<execspace_e::cuda>({}, grid),
                 dxinv = 1.f / grid.dx] __device__(auto pi) mutable {
                  using grid_t = RM_CVREF_T(grid);
                  auto pos = pars.pack<3>("pos", pi);

                  auto c = (pos * dxinv - 0.5);
                  typename ZenoPartition::table_t::key_t coord{};
                  for (int d = 0; d != 3; ++d)
                    coord[d] = lower_trunc(c[d]);
                  auto bno =
                      table.query(coord - (coord & (grid_t::side_length - 1)));
                  auto cno = grid_t::coord_to_cellid(
                      (coord & (grid_t::side_length - 1)));

                  auto vel = pars.pack<3>("vel", pi);
                  if ((vel(1) < -5.1 || vel(1) > -4.9)) {
                    printf("par[%d], [bi, ci]: (%d, %d). pos: %f, %f, %f; vel: "
                           "%f, %f, %f\n",
                           (int)pi, (int)bno, (int)cno, pos[0], pos[1], pos[2],
                           vel[0], vel[1], vel[2]);
                  }
                });
        puts("done g2p_vert_surface check");
        getchar();
#endif
        auto &eles = parObjPtr->getQuadraturePoints();
        if (parObjPtr->category == ZenoParticles::surface) {
          cudaPol(range(eles.size()),
                  [verts = proxy<execspace_e::cuda>({}, pars),
                   eles = proxy<execspace_e::cuda>({}, eles),
                   table = proxy<execspace_e::cuda>(partition),
                   grid = proxy<execspace_e::cuda>({}, grid), dt = stepDt,
                   dxinv = 1.f / grid.dx] __device__(size_t pi) mutable {
                    using mat2 = zs::vec<float, 2, 2>;
                    using mat3 = zs::vec<float, 3, 3>;
                    using grid_t = RM_CVREF_T(grid);
                    const auto Dinv = 4.f * dxinv * dxinv;
                    auto pos = eles.pack<3>("pos", pi);
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
                      auto vi = block.pack<3>(
                          "v", grid_t::coord_to_cellid(localIndex));

                      C += W * Dinv * dyadic_prod(vi, xixp);
                    }
                    auto skew = 0.5f * (C - C.transpose());
                    auto sym = 0.5f * (C + C.transpose());
                    C = skew + sym;
                    eles.tuple<9>("C", pi) = C;

                    // section 4.3
                    auto i0 = (int)eles("inds", 0, pi);
                    auto i1 = (int)eles("inds", 1, pi);
                    auto i2 = (int)eles("inds", 2, pi);

                    auto p0 = verts.pack<3>("pos", i0);
                    auto p1 = verts.pack<3>("pos", i1);
                    auto p2 = verts.pack<3>("pos", i2);
                    // pos
                    eles.tuple<3>("pos", pi) = (p0 + p1 + p2) / 3;
                    // vel
                    eles.tuple<3>("vel", pi) =
                        (verts.pack<3>("vel", i0) + verts.pack<3>("vel", i1) +
                         verts.pack<3>("vel", i2)) /
                        3;

                    // d
                    auto d_c1 = p1 - p0;
                    auto d_c2 = p2 - p0;
                    auto d_c3 = col(eles.pack<3, 3>("d", pi), 2);
// d_c3 += dt * (vGrad * d_c3);
#if 0
                    d_c3 += dt * (C * d_c3);
#else
                    d_c3 += (dt * C + 0.5 * dt * dt * C * C) * d_c3;
#endif

                    mat3 d{d_c1[0], d_c2[0], d_c3[0], d_c1[1], d_c2[1],
                           d_c3[1], d_c1[2], d_c2[2], d_c3[2]};
                    eles.tuple<9>("d", pi) = d;
                    // F
                    eles.tuple<9>("F", pi) = d * eles.pack<3, 3>("DmInv", pi);
                  });

#if 0
          cudaPol(
              Collapse{eles.size()},
              [eles = proxy<execspace_e::cuda>({}, eles),
               table = proxy<execspace_e::cuda>(partition),
               grid = proxy<execspace_e::cuda>({}, grid),
               dxinv = 1.f / grid.dx] __device__(auto pi) mutable {
                using grid_t = RM_CVREF_T(grid);
                auto pos = eles.pack<3>("pos", pi);

                auto c = (pos * dxinv - 0.5);
                typename ZenoPartition::table_t::key_t coord{};
                for (int d = 0; d != 3; ++d)
                  coord[d] = lower_trunc(c[d]);
                auto bno =
                    table.query(coord - (coord & (grid_t::side_length - 1)));
                auto cno = grid_t::coord_to_cellid(
                    (coord & (grid_t::side_length - 1)));

                auto vel = eles.pack<3>("vel", pi);
                if ((vel(1) < -5.1 || vel(1) > -4.9)) {
                  printf("ele[%d], [bi, ci]: (%d, %d). pos: %f, %f, %f; vel: "
                         "%f, %f, %f\n",
                         (int)pi, (int)bno, (int)cno, pos[0], pos[1], pos[2],
                         vel[0], vel[1], vel[2]);
                }
              });
          puts("done g2p_ele_surface check");
          getchar();
#endif
        } // case: surface
      }   // end mesh particle g2p
    }
    fmt::print(fg(fmt::color::cyan), "done executing ZSGridToZSParticle\n");
    set_output("ZSParticles", get_input("ZSParticles"));
  }
};

ZENDEFNODE(ZSGridToZSParticle,
           {
               {"ZSGrid", "ZSPartition", "ZSParticles", "dt"},
               {"ZSParticles"},
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
      constexpr auto k = 100.f;
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

struct TransformZSLevelSet : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing TransformZSLevelSet\n");
    auto zsls = get_input<ZenoLevelSet>("ZSLevelSet");
    auto &ls = zsls->getLevelSet();

    using namespace zs;
    using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
    // translation
    if (has_input("translation")) {
      auto b = get_input<NumericObject>("translation")->get<vec3f>();
      match(
          [&b](basic_ls_t &basicLs) {
            match(
                [b](std::shared_ptr<typename basic_ls_t::clspls_t> lsPtr) {
                  lsPtr->translate(zs::vec<float, 3>{b[0], b[1], b[2]});
                },
                [b](std::shared_ptr<typename basic_ls_t::ccspls_t> lsPtr) {
                  lsPtr->translate(zs::vec<float, 3>{b[0], b[1], b[2]});
                },
                [b](std::shared_ptr<typename basic_ls_t::sgspls_t> lsPtr) {
                  lsPtr->translate(zs::vec<float, 3>{b[0], b[1], b[2]});
                },
                [](auto &lsPtr) {
                  auto msg = get_var_type_str(*lsPtr);
                  throw std::runtime_error(fmt::format(
                      "levelset of type [{}] cannot be transformed yet.", msg));
                })(basicLs._ls);
          },
          [](auto &ls) {
            auto msg = get_var_type_str(ls);
            throw std::runtime_error(
                fmt::format("levelset of special type [{}] are const-.", msg));
          })(ls);
    }

    // scale
    if (has_input("scaling")) {
      auto s = get_input<NumericObject>("scaling")->get<float>();
      match(
          [&s](basic_ls_t &basicLs) {
            match(
                [s](std::shared_ptr<typename basic_ls_t::clspls_t> lsPtr) {
                  lsPtr->scale(s);
                },
                [s](std::shared_ptr<typename basic_ls_t::ccspls_t> lsPtr) {
                  lsPtr->scale(s);
                },
                [s](std::shared_ptr<typename basic_ls_t::sgspls_t> lsPtr) {
                  lsPtr->scale(s);
                },
                [](auto &lsPtr) {
                  auto msg = get_var_type_str(*lsPtr);
                  throw std::runtime_error(fmt::format(
                      "levelset of type [{}] cannot be transformed yet.", msg));
                })(basicLs._ls);
          },
          [](auto &ls) {
            auto msg = get_var_type_str(ls);
            throw std::runtime_error(
                fmt::format("levelset of special type [{}] are const-.", msg));
          })(ls);
    }
    // rotation
    if (has_input("eulerXYZ")) {
      auto yprAngles = get_input<NumericObject>("eulerXYZ")->get<vec3f>();
      auto rot = zs::Rotation<float, 3>{yprAngles[0], yprAngles[1],
                                        yprAngles[2], zs::degree_c, zs::ypr_c};
      match(
          [&rot](basic_ls_t &basicLs) {
            match(
                [rot](std::shared_ptr<typename basic_ls_t::clspls_t> lsPtr) {
                  lsPtr->rotate(rot.transpose());
                },
                [rot](std::shared_ptr<typename basic_ls_t::ccspls_t> lsPtr) {
                  lsPtr->rotate(rot.transpose());
                },
                [rot](std::shared_ptr<typename basic_ls_t::sgspls_t> lsPtr) {
                  lsPtr->rotate(rot.transpose());
                },
                [](auto &lsPtr) {
                  auto msg = get_var_type_str(*lsPtr);
                  throw std::runtime_error(fmt::format(
                      "levelset of type [{}] cannot be transformed yet.", msg));
                })(basicLs._ls);
          },
          [](auto &ls) {
            auto msg = get_var_type_str(ls);
            throw std::runtime_error(
                fmt::format("levelset of special type [{}] are const-.", msg));
          })(ls);
    }

    fmt::print(fg(fmt::color::cyan), "done executing TransformZSLevelSet\n");
    set_output("ZSLevelSet", zsls);
  }
};
// refer to nodes/prim/TransformPrimitive.cpp
ZENDEFNODE(TransformZSLevelSet,
           {
               {"ZSLevelSet", "translation", "eulerXYZ", "scaling"},
               {"ZSLevelSet"},
               {},
               {"MPM"},
           });

struct ZSBoundaryPrimitiveToZSGrid : INode {
  void p2g_momentum(zs::CudaExecutionPolicy &cudaPol,
                    const typename ZenoParticles::particles_t &pars,
                    const typename ZenoPartition::table_t &partition,
                    typename ZenoGrid::grid_t &grid,
                    bool includeNormal = false) {
    using namespace zs;

    cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 table = proxy<execspace_e::cuda>(partition),
                                 grid = proxy<execspace_e::cuda>({}, grid),
                                 dxinv = 1.f / grid.dx,
                                 includeNormal] __device__(size_t pi) mutable {
      using grid_t = RM_CVREF_T(grid);
      const auto Dinv = 4.f * dxinv * dxinv;
      auto pos = pars.pack<3>("pos", pi);
      auto vel = pars.pack<3>("vel", pi);
      auto mass = pars("mass", pi);
      // auto vol = pars("vol", pi);
      auto nrm = pars.pack<3>("nrm", pi);

      auto arena =
          make_local_arena<grid_e::collocated, kernel_e::linear>(grid.dx, pos);

      for (auto loc : arena.range()) {
        auto coord = arena.coord(loc);
        auto localIndex = coord & (grid_t::side_length - 1);
        auto blockno = table.query(coord - localIndex);
        if (blockno < 0)
          printf("THE HELL!");
        auto block = grid.block(blockno);
        auto W = arena.weight(loc);
        const auto cellid = grid_t::coord_to_cellid(localIndex);
        atomic_add(exec_cuda, &block("m", cellid), mass * W);
        for (int d = 0; d != 3; ++d)
          atomic_add(exec_cuda, &block("v", d, cellid), W * mass * vel[d]);
        if (includeNormal)
          for (int d = 0; d != 3; ++d)
            atomic_add(exec_cuda, &block("nrm", d, cellid), nrm[d]);
      }
    });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing ZSBoundaryPrimitiveToZSGrid\n");

    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);

    if (zsgrid->transferScheme != "boundary")
      throw std::runtime_error("grid is not of boundary type!");

    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();
      auto &eles = parObjPtr->getQuadraturePoints();
      if (!pars.hasProperty("nrm") || !eles.hasProperty("nrm"))
        throw std::runtime_error(
            "boundary primitive does not have normal channel!");
      p2g_momentum(cudaPol, pars, partition, grid, false);
      p2g_momentum(cudaPol, eles, partition, grid, true);
      fmt::print("[boundary particle p2g] dx: {}, npars: {}, neles: {}\n",
                 grid.dx, pars.size(), eles.size());
      // fmt::print("p2g boundary iterating par: {}\n", (void
      // *)parObjPtr.get());
    }

    fmt::print(fg(fmt::color::cyan),
               "done executing ZSBoundaryPrimitiveToZSGrid\n");
    set_output("ZSGrid", zsgrid);
  }
};

ZENDEFNODE(ZSBoundaryPrimitiveToZSGrid,
           {
               {"ZSParticles", "ZSPartition", "ZSGrid"},
               {"ZSGrid"},
               {},
               {"MPM"},
           });

struct ApplyWindImpulseOnZSGrid : INode {
  template <typename VelSplsViewT>
  void computeWindImpulse(zs::CudaExecutionPolicy &cudaPol, float windDragCoeff,
                          float windDensity, VelSplsViewT velLs,
                          const typename ZenoParticles::particles_t &pars,
                          const typename ZenoParticles::particles_t &eles,
                          const typename ZenoPartition::table_t &partition,
                          typename ZenoGrid::grid_t &grid, float dt) {
    using namespace zs;
    cudaPol(range(eles.size()),
            [windDragCoeff, windDensity, velLs,
             pars = proxy<execspace_e::cuda>({}, pars), // for normal compute
             eles = proxy<execspace_e::cuda>({}, eles),
             table = proxy<execspace_e::cuda>(partition),
             grid = proxy<execspace_e::cuda>({}, grid),
             Dinv = 4.f / grid.dx / grid.dx, dt] __device__(size_t ei) mutable {
              using grid_t = RM_CVREF_T(grid);
              zs::vec<float, 3> n{};
              float area{};
              {
                auto p0 = pars.pack<3>("pos", (int)eles("inds", 0, ei));
                auto p1 = pars.pack<3>("pos", (int)eles("inds", 1, ei));
                auto p2 = pars.pack<3>("pos", (int)eles("inds", 2, ei));
                auto cp = (p1 - p0).cross(p2 - p0);
                area = cp.length();
                n = cp / area;
                area *= 0.5f;
              }
              auto pos = eles.pack<3>("pos", ei);
              auto windVel = velLs.getMaterialVelocity(pos);

              auto vel = eles.pack<3>("vel", ei);
              auto vrel = windVel - vel;
              float vnSignedLength = n.dot(vrel);
              auto vn = n * vnSignedLength;
              auto vt = vrel - vn; // tangent
              auto windForce =
                  windDensity * area * zs::abs(vnSignedLength) * vn +
                  windDragCoeff * area * vt;
              auto fdt = windForce * dt;

              auto arena =
                  make_local_arena<grid_e::collocated, kernel_e::quadratic>(
                      grid.dx, pos);

              for (auto loc : arena.range()) {
                auto coord = arena.coord(loc);
                auto localIndex = coord & (grid_t::side_length - 1);
                auto blockno = table.query(coord - localIndex);
                if (blockno < 0)
                  printf("THE HELL!");
                auto block = grid.block(blockno);
                auto W = arena.weight(loc);
                const auto cellid = grid_t::coord_to_cellid(localIndex);
                for (int d = 0; d != 3; ++d)
                  atomic_add(exec_cuda, &block("v", d, cellid), W * fdt[d]);
              }
            });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing ApplyWindImpulseOnZSGrid\n");

    using namespace zs;

    // this could possibly be the same staggered velocity field too
    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();

    auto velZsField = get_input<ZenoLevelSet>("ZSVelField");
    const auto &velField = velZsField->getBasicLevelSet()._ls;

    auto stepDt = get_input2<float>("dt");
    auto windDrag = get_input2<float>("windDrag");
    auto windDensity = get_input2<float>("windDensity");

    match([&](const auto &velLsPtr) {
      auto cudaPol = cuda_exec().device(0);
      for (auto &&parObjPtr : parObjPtrs) {
        auto &pars = parObjPtr->getParticles();
        if (parObjPtr->category == ZenoParticles::surface ||
            parObjPtr->category == ZenoParticles::tracker) {
          auto &eles = parObjPtr->getQuadraturePoints();
          computeWindImpulse(cudaPol, windDrag, windDensity,
                             get_level_set_view<execspace_e::cuda>(velLsPtr),
                             pars, eles, partition, grid, stepDt);
        }
      }
    })(velField);

    fmt::print(fg(fmt::color::cyan),
               "done executing ApplyWindImpulseOnZSGrid\n");
    set_output("ZSGrid", zsgrid);
  }
};

ZENDEFNODE(ApplyWindImpulseOnZSGrid, {
                                         {"ZSParticles",
                                          "ZSVelField",
                                          "ZSPartition",
                                          "ZSGrid",
                                          {"float", "windDrag", "0"},
                                          {"float", "windDensity", "1"},
                                          {"float", "dt", "0.1"}},
                                         {"ZSGrid"},
                                         {},
                                         {"MPM"},
                                     });

} // namespace zeno