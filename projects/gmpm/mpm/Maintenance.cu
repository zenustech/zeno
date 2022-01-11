#include "../Utils.hpp"
#include "Structures.hpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"
#include <zeno/types/DictObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

struct ComputeParticleVolume : INode {
  void apply() override {
    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();

    auto buckets = std::make_shared<ZenoIndexBuckets>();
    auto &ibs = buckets->get();

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);
    bool first = true;

    for (auto &&parObjPtr : parObjPtrs)
      if (parObjPtr->category == ZenoParticles::mpm) {
        auto &pars = parObjPtr->getParticles();
        spatial_hashing(cudaPol, pars, grid.dx, ibs, first, true);
        first = false;
      }

    for (auto &&parObjPtr : parObjPtrs) {
      if (parObjPtr->category != ZenoParticles::mpm)
        continue;
      auto &pars = parObjPtr->getParticles();
      cudaPol(range(pars.size()),
              [pars = proxy<execspace_e::cuda>({}, pars),
               ibs = proxy<execspace_e::cuda>(ibs),
               density = parObjPtr->getModel().density,
               cellVol =
                   grid.dx * grid.dx * grid.dx] __device__(size_t pi) mutable {
                auto pos = pars.template pack<3>("pos", pi);
                auto coord = ibs.bucketCoord(pos);
                const auto bucketNo = ibs.table.query(coord);
                // bucketNo should be > 0
                const auto cnt = ibs.counts[bucketNo];
                const auto vol = cellVol / cnt;
                pars("vol", pi) = vol;
                pars("mass", pi) = density * vol;
              });
    }
    set_output("ZSParticles", get_input("ZSParticles"));
  }
};

ZENDEFNODE(ComputeParticleVolume,
           {
               {{"ZenoParticles", "ZSParticles"}, {"ZenoGrid", "ZSGrid"}},
               {{"ZenoParticles", "ZSParticles"}},
               {},
               {"MPM"},
           });

struct PushOutZSParticles : INode {
  template <typename LsView>
  void pushout(zs::CudaExecutionPolicy &cudaPol,
               typename ZenoParticles::particles_t &pars, LsView lsv,
               float dis) {
    using namespace zs;
    cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars), lsv,
                                 eps = limits<float>::epsilon() * 128,
                                 dis] __device__(size_t pi) mutable {
      auto x = pars.pack<3>("pos", pi);
      bool updated = false;
      int cnt = 5;
#if 0
      if (pi < 10) {
        printf("par[%d] sd (%f). x(%f, %f, %f) towards %f\n", (int)pi,
               lsv.getSignedDistance(x), x[0], x[1], x[2], dis);
      }
#endif
      for (auto sd = lsv.getSignedDistance(x); sd < dis && cnt--;) {
        auto diff = x.zeros();
        for (int i = 0; i != 3; i++) {
          auto v1 = x;
          auto v2 = x;
          v1[i] = x[i] + eps;
          v2[i] = x[i] - eps;
          diff[i] = (lsv.getSignedDistance(v1) - lsv.getSignedDistance(v2)) /
                    (eps + eps);
        }
        // normal info is missing will also cause this
        if (math::near_zero(diff.l2NormSqr()))
          break;
        auto n = diff.normalized();
        x -= n * sd;
        auto newSd = lsv.getSignedDistance(x);
#if 0
        if (pi < 10)
          printf("%d rounds left, par[%d] sdf (%f)->(%f). x(%f, %f, %f), n(%f, "
                 "%f, %f)\n",
                 cnt, (int)pi, sd, newSd, x[0], x[1], x[2], n[0], n[1], n[2]);
#endif
        if (                                 // newSd < sd ||
            newSd - sd < 0.5f * zs::abs(sd)) // new position should be no deeper
          break;
        updated = true;
        sd = newSd;
      }
      if (updated)
        pars.tuple<3>("pos", pi) = x;
    });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing PushOutZSParticles\n");
    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
    using namespace zs;
    auto cudaPol = cuda_exec().device(0);
    auto zsls = get_input<ZenoLevelSet>("ZSLevelSet");
    auto dis = get_input2<float>("dis");

    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();
      using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
      using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
      using const_transition_ls_t =
          typename ZenoLevelSet::const_transition_ls_t;
      match(
          [&](basic_ls_t &ls) {
            match([&](const auto &lsPtr) {
              auto lsv = get_level_set_view<execspace_e::cuda>(lsPtr);
              pushout(cudaPol, pars, lsv, dis);
            })(ls._ls);
          },
          [&](const_sdf_vel_ls_t &ls) {
            match([&](auto lsv) {
              pushout(cudaPol, pars, SdfVelFieldView{lsv}, dis);
            })(ls.template getView<execspace_e::cuda>());
          },
          [&](const_transition_ls_t &ls) {
            match([&](auto fieldPair) {
              auto &fvSrc = std::get<0>(fieldPair);
              auto &fvDst = std::get<1>(fieldPair);
              pushout(cudaPol, pars,
                      TransitionLevelSetView{SdfVelFieldView{fvSrc},
                                             SdfVelFieldView{fvDst}},
                      dis);
            })(ls.template getView<execspace_e::cuda>());
          })(zsls->getLevelSet());
    }
    fmt::print(fg(fmt::color::cyan), "done executing PushOutZSParticles\n");
    set_output("ZSParticles", get_input("ZSParticles"));
  }
};
ZENDEFNODE(PushOutZSParticles,
           {
               {"ZSParticles", "ZSLevelSet", {"float", "dis", "0.01"}},
               {"ZSParticles"},
               {},
               {"MPM"},
           });

} // namespace zeno
