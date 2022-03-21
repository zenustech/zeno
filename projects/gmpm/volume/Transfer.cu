#include "../Structures.hpp"
#include "../Utils.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
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

static ZpcInitializer g_zpc_initializer{};

struct ZSParticleToZSLevelSet : INode {
  template <typename LsT>
  void p2g(zs::CudaExecutionPolicy &cudaPol,
           const typename ZenoParticles::particles_t &pars, LsT &ls) {
    using namespace zs;
    ls.append_channels(cudaPol, pars.getPropertyTags());
    cudaPol(range(pars.size()),
            [pars = proxy<execspace_e::cuda>({}, pars),
             lsv = proxy<execspace_e::cuda>(ls)] __device__(auto pi) mutable {
              // static_assert(std::is_integral_v<RM_CVREF_T(pi)>, "haha
              // gotcha");
              using ls_t = RM_CVREF_T(lsv);
              auto pos = pars.pack<3>("pos", pi); // this is required
              auto vol = pars("vol", pi);         // as weight
#if 0
              auto vel = pars.pack<3>("vel", pi);
              auto vol = pars("vol", pi);

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
#endif
            });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing ZSParticleToZSLevelSet\n");

    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
    auto zslevelset = get_input<ZenoLevelSet>("ZSLevelSet");
    auto &field = zslevelset->getBasicLevelSet()._ls;

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);

    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();

      match(
          [&](auto &lsPtr)
              -> std::enable_if_t<
                  is_spls_v<typename RM_CVREF_T(lsPtr)::element_type>> {
            auto &ls = *lsPtr;
            fmt::print("[par2ls] dx: {}, npars: {}\n", ls._grid.dx,
                       pars.size());
            p2g(cudaPol, pars, ls);
          },
          [](...) {
            throw std::runtime_error(
                "field to be transfered must be a sparse levelset!");
          })(field);
    }

    fmt::print(fg(fmt::color::cyan), "done executing ZSParticleToZSLevelSet\n");
    set_output("ZSLevelSet", std::move(zslevelset));
  }
};

ZENDEFNODE(ZSParticleToZSLevelSet, {
                                       {"ZSParticles", "ZSLevelSet"},
                                       {"ZSLevelSet"},
                                       {},
                                       {"Volume"},
                                   });

/// currently assume mesh resolution
struct PrimitiveToZSLevelSet : INode {
  using SpLsT = zs::SparseLevelSet<3>;
  using TV = zs::vec<float, 3>;
  using IV = zs::vec<int, 3>;
  void iterate(zs::CudaExecutionPolicy &cudaPol, int nvoxels,
               const zs::Vector<TV> &pos, SpLsT &ls, zs::Vector<IV> &mi,
               zs::Vector<IV> &ma) {
    using namespace zs;
    cudaPol(range(pos.size()), [pos = proxy<execspace_e::cuda>(pos),
                                mi = proxy<execspace_e::cuda>(mi),
                                ma = proxy<execspace_e::cuda>(ma),
                                ls = proxy<execspace_e::cuda>({}, ls),
                                nvoxels] __device__(int pi) mutable {
      using lsv_t = RM_CVREF_T(ls);
      auto x = pos[pi];
      auto coord = ls.worldToCell(x);
      /// [-nvoxels, nvoxels]
      auto cnt = nvoxels * 2 + 1;
      // time-consuming
      for (auto loc : ndrange<3>(cnt)) {
        auto c = coord + make_vec<int>(loc) - nvoxels;
        ls._table.insert(c - (c & (lsv_t::side_length - 1)));
      }

      for (int d = 0; d != 3; ++d) {
        atomic_max(exec_cuda, &ma[0][d], coord[d] + nvoxels);
        atomic_min(exec_cuda, &mi[0][d], coord[d] - nvoxels);
      }
    });
  }
  void p2g(zs::CudaExecutionPolicy &cudaPol, int nvoxels,
           const zs::Vector<TV> &pos, SpLsT &ls) {
    using namespace zs;
    cudaPol(range(pos.size()), [pos = proxy<execspace_e::cuda>(pos),
                                ls = proxy<execspace_e::cuda>({}, ls),
                                nvoxels] __device__(int pi) mutable {
      using lsv_t = RM_CVREF_T(ls);
      auto x = pos[pi];
      auto coord = ls.worldToCell(x);
      /// [-nvoxels, nvoxels]
      auto cnt = nvoxels * 2 + 1;
      // time-consuming
      for (auto loc : ndrange<3>(cnt)) {
        auto c = coord + make_vec<int>(loc) - nvoxels;
        auto cellcoord = c & (lsv_t::side_length - 1);
        auto block = ls._grid.block(ls._table.query(c - cellcoord));
        auto cellno = lsv_t::grid_view_t::coord_to_cellid(cellcoord);
        auto nodePos = ls.indexToWorld(c);
        constexpr float eps = limits<float>::epsilon();
        auto dis = zs::sqrt((x - nodePos).l2NormSqr() + eps);

        atomic_min(exec_cuda, &block("sdf", cellno), dis);
      }
    });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing PrimitiveToZSLevelSet\n");

    // primitive
    auto inParticles = get_input<PrimitiveObject>("prim");
    const auto &pos = inParticles->attr<vec3f>("pos");

    std::size_t numEles = 0;
    const auto &quads = inParticles->quads.values;
    const auto &tris = inParticles->tris.values;
    const auto &lines = inParticles->lines.values;
    if (quads.size())
      numEles = quads.size();
    else if (tris.size())
      numEles = tris.size();
    else if (lines.size())
      numEles = lines.size();

    using namespace zs;
    const auto dx = get_input2<float>("dx");
    Vector<TV> xs{pos.size(), memsrc_e::device, 0},
        elePos{numEles, memsrc_e::device, 0};
    zs::copy(zs::mem_device, (void *)xs.data(), (void *)pos.data(),
             sizeof(zeno::vec3f) * pos.size());
    auto cudaPol = cuda_exec().device(0);
    {
      if (quads.size()) {
        Vector<zs::vec<int, 4>> inds{numEles, memsrc_e::device, 0};
        zs::copy(zs::mem_device, (void *)inds.data(), (void *)quads.data(),
                 sizeof(zeno::vec4i) * numEles);
        cudaPol(
            range(elePos.size()),
            [elePos = proxy<execspace_e::cuda>(elePos),
             xs = proxy<execspace_e::cuda>(xs),
             inds = proxy<execspace_e::cuda>(inds)] __device__(int ei) mutable {
              const auto is = inds[ei];
              elePos[ei] =
                  (xs[is[0]] + xs[is[1]] + xs[is[2]] + xs[is[3]]) / 4.f;
            });
      } else if (tris.size()) {
        Vector<zs::vec<int, 3>> inds{numEles, memsrc_e::device, 0};
        zs::copy(zs::mem_device, (void *)inds.data(), (void *)tris.data(),
                 sizeof(zeno::vec3i) * numEles);
        cudaPol(
            range(elePos.size()),
            [elePos = proxy<execspace_e::cuda>(elePos),
             xs = proxy<execspace_e::cuda>(xs),
             inds = proxy<execspace_e::cuda>(inds)] __device__(int ei) mutable {
              const auto is = inds[ei];
              elePos[ei] = (xs[is[0]] + xs[is[1]] + xs[is[2]]) / 3.f;
            });
      } else if (lines.size()) {
        Vector<zs::vec<int, 2>> inds{numEles, memsrc_e::device, 0};
        zs::copy(zs::mem_device, (void *)inds.data(), (void *)lines.data(),
                 sizeof(zeno::vec2i) * numEles);
        cudaPol(
            range(elePos.size()),
            [elePos = proxy<execspace_e::cuda>(elePos),
             xs = proxy<execspace_e::cuda>(xs),
             inds = proxy<execspace_e::cuda>(inds)] __device__(int ei) mutable {
              const auto is = inds[ei];
              elePos[ei] = (xs[is[0]] + xs[is[1]]) / 2.f;
            });
      }
    }

    // 3 ^ (0.3f)
    const float thickness = get_input2<float>("thickness");
    const int nvoxels = (int)std::ceil(thickness * 1.45f / dx);
    auto zsspls = std::make_shared<ZenoLevelSet>();
    zsspls->getLevelSet() = ZenoLevelSet::basic_ls_t{
        std::make_shared<SpLsT>(dx, memsrc_e::device, 0)};
    SpLsT &ls = zsspls->getSparseLevelSet(zs::wrapv<SpLsT::category>{});

    // at most one block per particle
    ls._backgroundValue = dx * 2;
    ls._table =
        typename SpLsT::table_t{pos.size() + numEles, memsrc_e::device, 0};
    ls._table.reset(cudaPol, true);

    Vector<IV> mi{1, memsrc_e::device, 0}, ma{1, memsrc_e::device, 0};
    mi.setVal(IV::uniform(limits<int>::max()));
    ma.setVal(IV::uniform(limits<int>::lowest()));
    iterate(cudaPol, nvoxels, xs, ls, mi, ma);
    if (numEles)
      iterate(cudaPol, nvoxels, elePos, ls, mi, ma);
    ls._min = mi.getVal();
    ls._max = ma.getVal();

    const std::size_t nblocks = ls._table.size();
    fmt::print(
        "{} grid blocks. surface mesh ibox: [{}, {}, {}] - [{}, {}, {}]\n",
        nblocks, ls._min[0], ls._min[1], ls._min[2], ls._max[0], ls._max[1],
        ls._max[2]);

    ls._grid =
        typename SpLsT::grid_t{{{"sdf", 1}}, dx, nblocks, memsrc_e::device, 0};
    // clear grid
    cudaPol(Collapse{nblocks, ls.block_size},
            [grid = proxy<execspace_e::cuda>({}, ls._grid), thickness,
             backgroundValue = ls._backgroundValue] __device__(int bi,
                                                               int ci) mutable {
              grid("sdf", bi, ci) = thickness + backgroundValue;
            });
    // compute distance
    p2g(cudaPol, nvoxels, xs, ls);
    if (numEles)
      p2g(cudaPol, nvoxels, elePos, ls);
    // convert to sdf
    cudaPol(Collapse{nblocks, ls.block_size},
            [grid = proxy<execspace_e::cuda>({}, ls._grid), thickness,
             backgroundValue = ls._backgroundValue] __device__(int bi,
                                                               int ci) mutable {
              if (auto sdf = grid("sdf", bi, ci);
                  sdf < thickness) // touched by particles
                grid("sdf", bi, ci) = sdf - thickness;
              else
                grid("sdf", bi, ci) = backgroundValue;
            });
#if 0
    // check
    cudaPol(Collapse{nblocks, ls.block_size},
            [grid = proxy<execspace_e::cuda>({}, ls._grid),
             ls = proxy<execspace_e::cuda>({}, ls),
             thickness] __device__(int bi, int ci) mutable {
              auto x = ls.indexToWorld(bi, ci);
              if (auto sdf = grid("sdf", bi, ci);
                  sdf < 0) // touched by particles
                printf("grid(%d, %d) sdf: %f, at (%f, %f, %f)\n", bi, ci, sdf,
                       x[0], x[1], x[2]);
            });
#endif

    fmt::print(fg(fmt::color::cyan), "done executing PrimitiveToZSLevelSet\n");
    set_output("ZSLevelSet", zsspls);
  }
};

ZENDEFNODE(PrimitiveToZSLevelSet,
           {
               {"prim", {"float", "dx", "0.1"}, {"float", "thickness", "0.1"}},
               {"ZSLevelSet"},
               {},
               {"Volume"},
           });

} // namespace zeno