#include "../Structures.hpp"
#include "../Utils.hpp"

#include "zensim/container/Bvh.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/math/MathUtils.h"
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
      if (parObjPtr->category != ZenoParticles::bending)
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
                auto x = pars.template pack<3>("x", pi);
                auto c = (x * dxinv - 0.5);
                typename Partition::key_t coord{};
                for (int d = 0; d != 3; ++d)
                  coord[d] = lower_trunc(c[d]);
                table.insert(coord - (coord & (grid_t::side_length - 1)));
              });
      if (parObjPtr->isMeshPrimitive()) { // including tracker, but not bending
        auto &eles = parObjPtr->getQuadraturePoints();
        cudaPol(range(eles.size()),
                [eles = proxy<execspace_e::cuda>({}, eles),
                 table = proxy<execspace_e::cuda>(partition),
                 dxinv = 1.f / grid.dx] __device__(size_t ei) mutable {
                  auto x = eles.template pack<3>("x", ei);
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

struct SpringSystemTimeStepping : INode {
  using vec3 = zs::vec<float, 3>;
  using tiles_t = typename ZenoParticles::particles_t;
  // let verts above y=0.5 static
  void filter(zs::CudaExecutionPolicy &cudaPol, tiles_t &vertData,
              const zs::SmallString tagSrc, const zs::SmallString tagDst) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    cudaPol(range(vertData.size()), [data = proxy<space>({}, vertData), tagSrc,
                                     tagDst] __device__(int pi) mutable {
      auto v = vec3::zeros();
      if (data("x0", 1, pi) < 0.5)
        v = data.pack<3>(tagSrc, pi);
      data.tuple<3>(tagDst, pi) = v;
    });
  }
  float dot(zs::CudaExecutionPolicy &cudaPol, tiles_t &vertData,
            const zs::SmallString tag0, const zs::SmallString tag1) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    Vector<float> res{vertData.get_allocator(), 1};
    res.setVal(0);
    cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData), res = proxy<space>(res), tag0,
             tag1] __device__(int pi) mutable {
              auto v0 = data.pack<3>(tag0, pi);
              auto v1 = data.pack<3>(tag1, pi);
              atomic_add(exec_cuda, res.data(), v0.dot(v1));
            });
    return res.getVal();
  }
  float evalEps(zs::CudaExecutionPolicy &cudaPol, tiles_t &vertData, float dt) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    Vector<float> res{vertData.get_allocator(), 1};
    res.setVal(0);
    cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData),
             res = proxy<space>(res)] __device__(int pi) mutable {
              auto v = data.pack<3>("v", pi);
              auto dv = data.pack<3>("dv", pi);
              v += dv;
              atomic_add(exec_cuda, res.data(), v.dot(v));
            });
    auto yNorm = zs::sqrt(res.getVal(0) * dt);
    if (math::near_zero(yNorm))
      return zs::sqrt(limits<float>::epsilon());
    res.setVal(0);
    cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData), res = proxy<space>(res),
             dt] __device__(int pi) mutable {
              auto x0 = data.pack<3>("x0", pi);
              atomic_add(exec_cuda, res.data(), x0.dot(x0));
            });
    auto xNorm = zs::sqrt(res.getVal());
    return zs::sqrt((1 + xNorm) * limits<float>::epsilon()) / yNorm;
  }
  float computeSpringEnergy(zs::CudaExecutionPolicy &cudaPol,
                            const tiles_t &springs, tiles_t &vertData,
                            float alpha, float dt) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    Vector<double> Esum{vertData.get_allocator(), 1};
    Esum.setVal(0);
    cudaPol(range(springs.size()), [Esum = proxy<space>(Esum),
                                    eles = proxy<space>({}, springs),
                                    data = proxy<space>({}, vertData), dt,
                                    alpha] __device__(int ei) mutable {
      auto inds = eles.pack<2>("inds", ei).reinterpret_bits<int>();
      auto v0 = data.pack<3>("v", inds[0]);
      auto dv0 =
          data.pack<3>("dv", inds[0]) + data.pack<3>("p", inds[0]) * alpha;
      v0 += dv0;
      auto x0 = data.pack<3>("x0", inds[0]) + v0 * dt;

      auto v1 = data.pack<3>("v", inds[1]);
      auto dv1 =
          data.pack<3>("dv", inds[1]) + data.pack<3>("p", inds[1]) * alpha;
      v1 += dv1;
      auto x1 = data.pack<3>("x0", inds[1]) + v1 * dt;

      auto k = eles("k", ei);
      auto rl = eles("rl", ei);

      auto l = (x1 - x0).norm();
      auto E = 0.5 * k * zs::sqr(l - rl);
      atomic_add(exec_cuda, &Esum[0], E);
    });
    return Esum.getVal();
  }
  void computeSpringForce(zs::CudaExecutionPolicy &cudaPol,
                          const tiles_t &springs, tiles_t &vertData,
                          const zs::SmallString vtag,
                          const zs::SmallString ftag, float dt, float eps,
                          bool clear = false) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (clear)
      cudaPol(range(vertData.size()),
              [data = proxy<space>({}, vertData), ftag] __device__(
                  int pi) mutable { data.tuple<3>(ftag, pi) = vec3::zeros(); });
    cudaPol(range(springs.size()), [eles = proxy<space>({}, springs),
                                    data = proxy<space>({}, vertData), vtag,
                                    ftag, eps, dt] __device__(int ei) mutable {
      auto inds = eles.pack<2>("inds", ei).reinterpret_bits<int>();
      auto dv0 = data.pack<3>(vtag, inds[0]) * eps;
      auto x0 = data.pack<3>("x0", inds[0]);
      auto dv1 = data.pack<3>(vtag, inds[1]) * eps;
      auto x1 = data.pack<3>("x0", inds[1]);

      auto k = eles("k", ei);
      auto rl = eles("rl", ei);

      auto fdir0 = (x1 - x0).normalized();
      x0 += dv0 * dt;
      x1 += dv1 * dt;
      auto fdir = x1 - x0;
      auto dist = fdir.norm();

      vec3 f{};
      if (dist < rl) // being compressed, project
        f = k * (fdir.dot(fdir0) - rl) * fdir0;
      else
        f = k * (dist - rl) * (fdir / dist);

      for (int d = 0; d != 3; ++d) {
        atomic_add(exec_cuda, &data(ftag, d, inds[0]), f[d]);
        atomic_add(exec_cuda, &data(ftag, d, inds[1]), -f[d]);
      }
    });
  }
  void computeCollisionForce(zs::CudaExecutionPolicy &cudaPol,
                             const tiles_t &springs, tiles_t &vertData,
                             const zs::SmallString vtag,
                             const zs::SmallString ftag, float dt, float eps,
                             bool clear = false) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    if (clear)
      cudaPol(range(vertData.size()),
              [data = proxy<space>({}, vertData), ftag] __device__(
                  int pi) mutable { data.tuple<3>(ftag, pi) = vec3::zeros(); });
    auto numSprings = springs.size();
    using lbvh_t = zs::LBvh<3>;
    using bv_t = typename lbvh_t::Box;
    lbvh_t lbvh{};
    zs::Vector<bv_t> bvs{vertData.get_allocator(), numSprings};
#if 0
    float thickness = 0.0002;
    cudaPol(range(numSprings),
            [eles = proxy<space>({}, springs), bvs = proxy<space>(bvs),
             data = proxy<space>({}, vertData), vtag, eps, dt,
             thickness] __device__(int ei) mutable {
              auto inds = eles.pack<2>("inds", ei).reinterpret_bits<int>();
              auto dv0 = data.pack<3>(vtag, inds[0]) * eps;
              auto x0 = data.pack<3>("x0", inds[0]);
              auto dv1 = data.pack<3>(vtag, inds[1]) * eps;
              auto x1 = data.pack<3>("x0", inds[1]);
              x0 += dv0 * dt;
              x1 += dv1 * dt;
              auto [mi, ma] = get_bounding_box(x0, x1);
              bv_t bv{mi, ma};
              // thicken the box
              bv._min -= thickness * 3;
              bv._max += thickness * 3;
              bvs[ei] = bv;
            });
    lbvh.build(cudaPol, bvs);
    cudaPol(range(numSprings), [eles = proxy<space>({}, springs),
                                data = proxy<space>({}, vertData),
                                bvh = proxy<space>(lbvh), vtag, ftag, eps, dt,
                                thickness] __device__(int ei) mutable {
      auto getMovedVerts = [&](const zs::vec<int, 2> &inds) {
        auto dv0 = data.pack<3>(vtag, inds[0]) * dt;
        auto x0 = data.pack<3>("x0", inds[0]) + dv0 * eps;
        auto dv1 = data.pack<3>(vtag, inds[1]) * dt;
        auto x1 = data.pack<3>("x0", inds[1]) + dv1 * eps;

        x0 += dv0 * dt;
        x1 += dv1 * dt;
        return zs::make_tuple(x0.cast<double>(), x1.cast<double>());
      };
      auto inds = eles.pack<2>("inds", ei).reinterpret_bits<int>();
      auto [x0, x1] = getMovedVerts(inds);
      double areaWeight = (x1 - x0).norm() * thickness;

      using bvh_t = RM_CVREF_T(bvh);
      using bv_t = typename bvh_t::bv_t;
      using Ti = typename bvh_t::index_t;
      bv_t bv = get_bounding_box(x0, x1);

      const Ti numNodes = bvh._numNodes;
      Ti node = 0;
      while (node != -1 && node != numNodes) {
        Ti level = bvh._levels[node];
        // level and node are always in sync
        for (; level; --level, ++node)
          if (!overlaps(bvh.getNodeBV(node), bv))
            break;
        // leaf node check
        if (level == 0) {
          if (overlaps(bvh.getNodeBV(node), bv)) {
            auto oeid = bvh._auxIndices[node];
            auto oe_inds = eles.pack<2>("inds", oeid).reinterpret_bits<int>();
            if ((oe_inds[0] != inds[0] && oe_inds[0] != inds[1]) &&
                (oe_inds[1] != inds[0] && oe_inds[1] != inds[1])) {
              auto [o_x0, o_x1] = getMovedVerts(oe_inds);
              auto dist2 = zs::dist2_ee(x0, x1, o_x0, o_x1);
              //
              constexpr double xi = 1e-4;
              constexpr double xi2 = xi * xi;
              if (dist2 < xi2)
                printf("penetrated already...");
              constexpr double dHat = 1e-3;
              constexpr double activeGap2 = dHat * dHat;
              constexpr double kappa = 1e3;
              double o_areaWeight = (o_x1 - o_x0).norm() * thickness;
              double aw = (areaWeight + o_areaWeight) / 4;
              //
              auto grad = zs::dist_grad_ee(x0, x1, o_x0, o_x1);
              grad *= barrier_gradient(dist2 - xi2, activeGap2, kappa);
              grad *= aw * dHat;
              for (int vi = 0; vi != 2; ++vi) {
                auto f = row(grad, vi);
                for (int d = 0; d != 3; ++d)
                  atomic_add(exec_cuda, &data(ftag, d, inds[vi]), (float)f[d]);
              }
              for (int vi = 0; vi != 2; ++vi) {
                auto f = row(grad, vi + 2);
                for (int d = 0; d != 3; ++d)
                  atomic_add(exec_cuda, &data(ftag, d, oe_inds[vi]),
                             (float)f[d]);
              }
            }
          }
          node++;
        } else // separate at internal nodes
          node = bvh._auxIndices[node];
      }
    });
#endif
  }
  void SAp(zs::CudaExecutionPolicy &cudaPol, const tiles_t &verts,
           const tiles_t &springs, tiles_t &vertData, float dt, float eps) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    computeSpringForce(cudaPol, springs, vertData, "p", "f", dt, eps, true);
    computeCollisionForce(cudaPol, springs, vertData, "p", "f", dt, eps);
    cudaPol(range(vertData.size()),
            [verts = proxy<space>({}, verts), data = proxy<space>({}, vertData),
             dt, eps] __device__(int pi) mutable {
              auto m = verts("m", pi);
              auto f0 = data.pack<3>("f0", pi);
              auto f = data.pack<3>("f", pi);
              auto p = data.pack<3>("p", pi);
              data.tuple<3>("s", pi) = p - dt / m * (f - f0) / eps;
            });
    filter(cudaPol, vertData, "s", "s");
  }

  void apply() override {
    using namespace zs;
    fmt::print(fg(fmt::color::green),
               "begin executing SpringSystemTimeStepping\n");
    vec3 accel{vec3::zeros()};
    if (has_input("Accel"))
      accel = vec3::from_array(
          get_input<NumericObject>("Accel")->get<zeno::vec3f>());
    auto dt = get_input2<float>("dt");
    auto zssprings = get_input<ZenoParticles>("ZSSprings");
    auto &verts = zssprings->getParticles();
    auto &springs = zssprings->getQuadraturePoints();

    constexpr auto space = execspace_e::cuda;
    tiles_t vertData{verts.get_allocator(),
                     {{"x0", 3},
                      {"dv", 3},
                      {"v", 3},
                      {"f", 3},
                      {"f0", 3},
                      {"b", 3},
                      {"r", 3},
                      {"s", 3},
                      {"p", 3}},
                     verts.size()};
    float deltaOld, deltaNew, delta0, alpha;

    auto cudaPol = zs::cuda_exec();
    auto numVerts = verts.size();
    auto numSprings = springs.size();

    // clear dv
    cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData)] __device__(int pi) mutable {
              data.tuple<3>("dv", pi) = vec3::zeros();
            });
    // f_initial
    cudaPol(range(numVerts),
            [verts = proxy<space>({}, verts),
             data = proxy<space>({}, vertData)] __device__(int pi) mutable {
              data.tuple<3>("x0", pi) = verts.pack<3>("x", pi);
              data.tuple<3>("v", pi) = verts.pack<3>("v", pi);
            });
    float eps = evalEps(cudaPol, vertData, dt);
    // float eps = 1e-4;
    fmt::print("initial eps: {}\n", eps);
#if 1
    computeSpringForce(cudaPol, springs, vertData, "v", "f0", 0, 0, true);
    computeCollisionForce(cudaPol, springs, vertData, "v", "f0", 0, 0);

    // f_forward(v0)
    computeSpringForce(cudaPol, springs, vertData, "v", "f", dt, eps, true);
    computeCollisionForce(cudaPol, springs, vertData, "v", "f", dt, eps);

    // compute b
    cudaPol(range(numVerts),
            [verts = proxy<space>({}, verts), data = proxy<space>({}, vertData),
             accel, dt, eps] __device__(int pi) mutable {
              auto m = verts("m", pi);
              auto f0 = data.pack<3>("f0", pi);
              auto f = data.pack<3>("f", pi);
              data.tuple<3>("b", pi) = dt * (accel + (f0 + (f - f0) / eps) / m);
            });

    // r = S(b)
    filter(cudaPol, vertData, "b", "r");
    filter(cudaPol, vertData, "r", "p");
    deltaNew = dot(cudaPol, vertData, "r", "p");
    delta0 = dot(cudaPol, vertData, "r", "r");
    auto tol = 1e-2;
    int iter = 0;
    // float Eprev = computeSpringEnergy(cudaPol, springs, vertData, 1, dt);
    // fmt::print("initial energy: {}\n", Eprev);
    while (deltaNew > tol * delta0 * tol &&
           deltaNew > limits<float>::epsilon() * 8) {
      // s = S(Ap)
      SAp(cudaPol, verts, springs, vertData, dt, eps);
      // alpha
      alpha = deltaNew / dot(cudaPol, vertData, "p", "s");
#if 0
      float E = computeSpringEnergy(cudaPol, springs, vertData, alpha, dt);
      for (; E > Eprev;) {
        alpha /= 2;
        E = computeSpringEnergy(cudaPol, springs, vertData, alpha, dt);
      }
#endif
      // dv += alpha * p, r -= alpha * s
      cudaPol(range(numVerts), [data = proxy<space>({}, vertData),
                                alpha] __device__(int pi) mutable {
        data.tuple<3>("dv", pi) =
            data.pack<3>("dv", pi) + alpha * data.pack<3>("p", pi);
        data.tuple<3>("r", pi) =
            data.pack<3>("r", pi) - alpha * data.pack<3>("s", pi);
      });
      deltaOld = deltaNew;
      deltaNew = dot(cudaPol, vertData, "r", "r");
      fmt::print("iteration [{}]: eps: {}; deltaOld {} -> deltaNew {}\n",
                 iter++, eps, deltaOld, deltaNew /*, E*/);
      //
      cudaPol(range(numVerts), [data = proxy<space>({}, vertData), deltaNew,
                                deltaOld] __device__(int pi) mutable {
        data.tuple<3>("p", pi) = data.pack<3>("r", pi) +
                                 (deltaNew / deltaOld) * data.pack<3>("p", pi);
      });
      filter(cudaPol, vertData, "p", "p");
      eps = evalEps(cudaPol, vertData, dt);
      // Eprev = E;
    }

    cudaPol(range(numVerts),
            [verts = proxy<space>({}, verts), data = proxy<space>({}, vertData),
             dt] __device__(int pi) mutable {
              auto v = data.pack<3>("v", pi) + data.pack<3>("dv", pi);
              verts.tuple<3>("v", pi) = v;
              verts.tuple<3>("x", pi) = verts.pack<3>("x", pi) + v * dt;
            });
#elif 0
    // explicit
    computeSpringForce(cudaPol, springs, vertData, "v", "f", dt, 1.f, true);
    cudaPol(range(numVerts),
            [verts = proxy<space>({}, verts), data = proxy<space>({}, vertData),
             accel, dt] __device__(int pi) mutable {
              auto m = verts("m", pi);
              auto v = verts.pack<3>("v", pi);
              auto dv = dt * (data.pack<3>("f", pi) / m + accel);
              if (verts("x", 1, pi) > 0.5)
                dv = vec3::zeros();
              v += dv;
              verts.tuple<3>("v", pi) = v;
              verts.tuple<3>("x", pi) = verts.pack<3>("x", pi) + v * dt;
            });
#endif

    fmt::print(fg(fmt::color::cyan),
               "done executing SpringSystemTimeStepping\n");
    getchar();
    set_output("ZSSprings", zssprings);
  }
};

ZENDEFNODE(SpringSystemTimeStepping,
           {
               {"ZSSprings", {"float", "dt", "0.01"}, "Accel"},
               {"ZSSprings"},
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
      auto d = eles.pack<3, 3>("d", pi);
      // hard code ftm
      constexpr auto gamma = 0.f;
      constexpr auto k = 40000.f;
      constexpr auto friction_coeff = 0.f;
      // constexpr auto friction_coeff = 0.17f;
      auto [Q, R] = math::gram_schmidt(d);
      auto apply = [&, &Q = Q, &R = R]() {
        d = Q * R;
        eles.tuple<9>("d", pi) = d;
        eles.tuple<9>("F", pi) = d * eles.pack<3, 3>("DmInv", pi);
      };
      if (gamma == 0.f) {
        R(0, 2) = R(1, 2) = 0;
        R(2, 2) = zs::min(R(2, 2), 1.f);
        apply();
      } else if (R(2, 2) > 1) {
        R(0, 2) = R(1, 2) = 0;
        R(2, 2) = 1;
        apply();
      } else if (R(2, 2) <= 0) { // inversion
        R(0, 2) = R(1, 2) = 0;
        R(2, 2) = zs::max(R(2, 2), -1.f);
        apply();
      } else if (R(2, 2) < 1) {
        auto rr = R(0, 2) * R(0, 2) + R(1, 2) * R(1, 2);
        auto r33_m_1 = R(2, 2) - 1;
        auto gamma_over_k = gamma / k;
        auto zz = friction_coeff * r33_m_1 * r33_m_1; // normal traction
        if (gamma_over_k * gamma_over_k * rr - zz * zz > 0) {
          auto scale = zz / (gamma_over_k * zs::sqrt(rr));
          R(0, 2) *= scale;
          R(1, 2) *= scale;
          apply();
        }
      }
    });
  }
  void return_mapping_curve(zs::CudaExecutionPolicy &cudaPol,
                            const zs::StvkWithHencky<float> &stvkModel,
                            typename ZenoParticles::particles_t &eles) const {
    using namespace zs;
    bool materialParamOverride =
        eles.hasProperty("mu") && eles.hasProperty("lam");
    // drucker prager for stvk elastic model
    // ref: libwetcloth, Jiang 2017
    cudaPol(range(eles.size()),
            [eles = proxy<execspace_e::cuda>({}, eles), stvkModel = stvkModel,
             materialParamOverride] __device__(size_t pi) mutable {
              // hard code ftm
              constexpr auto gamma = 10.f;
              constexpr auto alpha = 0.f;
              constexpr auto beta = 0.f;
              constexpr auto alpha_tangent = 0.f;
              constexpr auto cohesion = 0.f; // no cohesion ftm
              bool projected = false;
              if (materialParamOverride) {
                stvkModel.mu = eles("mu", pi);
                stvkModel.lam = eles("lam", pi);
              }
              auto d = eles.pack<3, 3>("d", pi);
              auto [Q, R] = math::gram_schmidt(d);

              using vec2 = zs::vec<float, 2>;
              using mat2 = zs::vec<float, 2, 2>;

              mat2 R_hat{R(1, 1), R(1, 2), R(2, 1), R(2, 2)};
              auto [U, S, V] = math::qr_svd(R_hat);
              auto eps =
                  S.abs().max(limits<float>::epsilon() * 128).log() - cohesion;
              auto eps_trace = eps.sum() /*+ logJp*/;
              if (eps_trace < 0) {
                auto eps_hat = eps - 0.5f * eps_trace;
                auto eps_hat_norm = eps_hat.norm();
                auto dgp = eps_hat_norm + (stvkModel.mu + stvkModel.lam) /
                                              stvkModel.mu * eps_trace * alpha;
                if (eps_hat_norm < limits<float>::epsilon())
                  eps = eps.zeros() + cohesion;
                else
                  eps = eps - dgp / eps_hat_norm * eps_hat + cohesion;
              } else {
                eps = eps.zeros() + cohesion;
              }
              S = eps.exp();
              auto R2 = diag_mul(U, S) * V.transpose();
              R(1, 1) = R2(0, 0);
              R(1, 2) = R2(0, 1);
              R(2, 1) = R2(1, 0);
              R(2, 2) = R2(1, 1);

              auto tau_hat = stvkModel.first_piola(R2) * R2.transpose();
              auto p_cohesion =
                  (stvkModel.mu * 2 + stvkModel.lam * 2) * cohesion;
              auto p = zs::min(trace(tau_hat) / 2, p_cohesion);

              auto r = vec2{R(0, 1), R(0, 2)};
              auto gammaRr = gamma * (R2 * r).norm();
              auto f = gammaRr + alpha_tangent * p;

#if 1
              // Jiang
              if (gamma == 0.f) {
                R(0, 1) = R(0, 2) = 0;
              } else if (f > p_cohesion) {
                auto scale = (p_cohesion - alpha_tangent * p) / gammaRr;
                r *= scale;
                R(0, 1) = r(0);
                R(0, 2) = r(1);
              }
#else
              // Raymond
              const auto ff =
                  stvkModel.mu * zs::sqrt(sqr(R(0, 1)) + sqr(R(0, 2)));
              auto tmp = eps / S;
              const auto fn =
                  (2.0f * stvkModel.mu * tmp + stvkModel.lam * eps.sum() / S)
                      .norm() *
                  0.5f;
              if (ff > 0 && ff > fn * beta) {
                const auto scale = zs::min(1.f, beta * fn / ff);
                R(0, 1) *= scale;
                R(0, 2) *= scale;
              }
#endif

              d = Q * R;
              eles.tuple<9>("d", pi) = d;
              eles.tuple<9>("F", pi) = d * eles.pack<3, 3>("DmInv", pi);
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
        else if (parObjPtr->category == ZenoParticles::curve) {
          const auto &models = parObjPtr->getModel();
          match(
              [this, &eles, &cudaPol](const StvkWithHencky<float> &stvkModel) {
                // use drucker prager plasticity for friction handling
                return_mapping_curve(cudaPol, stvkModel, eles);
              },
              [](...) {
                // do nothing
              })(models.getElasticModel());
        }
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