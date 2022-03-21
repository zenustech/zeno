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
struct SurfaceMeshToSDF : INode {
  using SpLsT = zs::SparseLevelSet<3>;
  using TV = zs::vec<float, 3>;
  using IV = zs::vec<int, 3>;
  void iterate(zs::CudaExecutionPolicy &cudaPol, int nvoxels,
               const zs::Vector<TV> &pos, SpLsT &ls, zs::Vector<IV> &mi,
               zs::Vector<IV> &ma) {
    using namespace zs;
    cudaPol(range(pos.size()),
            [pos = proxy<execspace_e::cuda>(pos),
             mi = proxy<execspace_e::cuda>(mi),
             ma = proxy<execspace_e::cuda>(ma),
             ls = proxy<execspace_e::cuda>({}, ls)] __device__(int pi) mutable {
              auto x = pos[pi];

              for (int d = 0; d != 3; ++d) {
                atomic_max(exec_cuda, &ma[0][d], 0);
                atomic_min(exec_cuda, &mi[0][d], 0);
              }
            });
  }
  void p2g(zs::CudaExecutionPolicy &cudaPol, const zs::Vector<TV> &pos,
           SpLsT &) {
    ;
  }
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing SurfaceMeshToSDF\n");

    // primitive
    auto inParticles = get_input<PrimitiveObject>("prim");
    const auto &pos = inParticles->attr<vec3f>("pos");
    const auto &tris = inParticles->tris.values;

    using namespace zs;
    const auto dx = get_input2<float>("dx");
    Vector<TV> xs{pos.size(), memsrc_e::device, 0},
        elePos{tris.size(), memsrc_e::device, 0};
    zs::copy(zs::mem_device, (void *)xs.data(), (void *)pos.data(),
             sizeof(zeno::vec3f) * pos.size());
    Vector<zs::vec<int, 3>> inds{tris.size(), memsrc_e::device, 0};
    zs::copy(zs::mem_device, (void *)inds.data(), (void *)tris.data(),
             sizeof(zeno::vec3i) * tris.size());
    auto cudaPol = cuda_exec().device(0);
    cudaPol(range(elePos.size()),
            [elePos = proxy<execspace_e::cuda>(elePos),
             xs = proxy<execspace_e::cuda>(xs),
             inds = proxy<execspace_e::cuda>(inds)] __device__(int ei) mutable {
              const auto is = inds[ei];
              elePos[ei] = (xs[is[0]] + xs[is[1]] + xs[is[2]]) / 3.f;
            });

    // 3 ^ (0.3f)
    const float thickness = get_input2<float>("thickness");
    const int nvoxels = (int)std::ceil(thickness * 1.45f / dx);
    auto zsspls = std::make_shared<ZenoLevelSet>();
    zsspls->getLevelSet() = ZenoLevelSet::basic_ls_t{
        std::make_shared<SpLsT>(dx, memsrc_e::device, 0)};
    SpLsT &ls = zsspls->getSparseLevelSet(zs::wrapv<SpLsT::category>{});

    // at most one block per particle
    ls._backgroundValue = dx * 2;
    ls._table = typename SpLsT::table_t{pos.size(), memsrc_e::device, 0};
    Vector<IV> mi{}, ma{};
    mi.setVal(IV::uniform(limits<int>::max()));
    ma.setVal(IV::uniform(limits<int>::max()));
    iterate(cudaPol, nvoxels, xs, ls, mi, ma);
    iterate(cudaPol, nvoxels, elePos, ls, mi, ma);

    const std::size_t nblocks = ls._table.size();
    ls._grid =
        typename SpLsT::grid_t{{{"sdf", 1}}, dx, nblocks, memsrc_e::device, 0};
    // clear grid
    cudaPol(Collapse{},
            [grid = proxy<execspace_e::cuda>({}, ls._grid)] __device__(
                int bi, int ci) mutable { grid("sdf", bi, ci) = 0.f; });
    // compute distance
    p2g(cudaPol, xs, ls);
    p2g(cudaPol, elePos, ls);
    // convert to sdf
    cudaPol(Collapse{}, [grid = proxy<execspace_e::cuda>({}, ls._grid),
                         thickness] __device__(int bi, int ci) mutable {
      grid("sdf", bi, ci) = grid("sdf", bi, ci) - thickness;
    });

    fmt::print(fg(fmt::color::cyan), "done executing SurfaceMeshToSDF\n");
    set_output("ZSLevelSet", std::move(zsspls));
  }
};

ZENDEFNODE(SurfaceMeshToSDF,
           {
               {"prim", {"float", "dx", "0.1"}, {"float", "thickness", "0.1"}},
               {"ZSLevelSet"},
               {},
               {"Volume"},
           });

} // namespace zeno