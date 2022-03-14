#include "../Structures.hpp"
#include "../Utils.hpp"
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

} // namespace zeno