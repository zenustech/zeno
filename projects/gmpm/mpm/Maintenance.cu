#include "../Structures.hpp"
#include "../Utils.hpp"

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
      for (auto sd = lsv.getSignedDistance(x), sdabs = zs::abs(sd);
           sd < dis && cnt--;) {
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
        x += n * sdabs;
        auto newSd = lsv.getSignedDistance(x);
#if 0
        if (pi < 10)
          printf("%d rounds left, par[%d] sdf (%f)->(%f). x(%f, %f, %f), n(%f, "
                 "%f, %f)\n",
                 cnt, (int)pi, sd, newSd, x[0], x[1], x[2], n[0], n[1], n[2]);
#endif
        if (                           // newSd < sd ||
            newSd - sd < 0.5f * sdabs) // new position should be no deeper
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
              auto lsv = zs::get_level_set_view<execspace_e::cuda>(lsPtr);
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
                                             SdfVelFieldView{fvDst}, ls._stepDt,
                                             ls._alpha},
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

struct RefineMeshParticles : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing RefineMeshParticles\n");

    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);

    /// the biggest distance among particles should be no greater than 'dx'
    auto dx = get_input2<float>("dx");

    for (auto &&parObjPtr : parObjPtrs) {
      if (parObjPtr->prim.get() == nullptr ||
          parObjPtr->category == ZenoParticles::mpm)
        continue;

      if (parObjPtr->category == ZenoParticles::surface) {
        auto &model = parObjPtr->getModel();
        auto &pars = parObjPtr->getParticles();
        auto &eles = parObjPtr->getQuadraturePoints();

        Vector<int> vertCnt{1, memsrc_e::device, 0},
            eleCnt{1, memsrc_e::device, 0};
        Vector<int> vertOffsets{1, memsrc_e::device, 0},
            eleOffsets{1, memsrc_e::device, 0};
        int prevVertCnt{}, prevEleCnt{};

        auto probeSize = [&]() {
          vertCnt.setVal(pars.size());
          eleCnt.setVal(eles.size());
          prevVertCnt = pars.size();
          prevEleCnt = eles.size();
          vertOffsets.resize(prevEleCnt);
          eleOffsets.resize(prevEleCnt);
          cudaPol(range(prevEleCnt),
                  [eles = proxy<execspace_e::cuda>({}, eles),
                   pars = proxy<execspace_e::cuda>({}, pars),
                   vertCnt = proxy<execspace_e::cuda>(vertCnt),
                   eleCnt = proxy<execspace_e::cuda>(eleCnt),
                   vertOffsets = proxy<execspace_e::cuda>(vertOffsets),
                   eleOffsets = proxy<execspace_e::cuda>(eleOffsets),
                   dx] __device__(int ei) mutable {
                    /// inds, xs
                    int inds[3] = {(int)eles("inds", 0, ei),
                                   (int)eles("inds", 1, ei),
                                   (int)eles("inds", 2, ei)};
                    zs::vec<float, 3> xs[3]{pars.pack<3>("pos", inds[0]),
                                            pars.pack<3>("pos", inds[1]),
                                            pars.pack<3>("pos", inds[2])};
                    auto area =
                        (xs[1] - xs[0]).cross(xs[2] - xs[0]).norm() * 0.5f;

                    // not the ideal heuristic
                    if (area > dx * dx * 0.5f) {
                      vertOffsets[ei] =
                          atomic_add(exec_cuda, vertCnt.data(), 1);
                      eleOffsets[ei] = atomic_add(exec_cuda, eleCnt.data(), 2);
                    } else {
                      vertOffsets[ei] = -1;
                      eleOffsets[ei] = -1;
                    }
                  });
          fmt::print("verts from {} to {}, {} added\n", prevVertCnt,
                     vertCnt.getVal(), vertCnt.getVal() - prevVertCnt);
          fmt::print("eles from {} to {}, {} added\n", prevEleCnt,
                     eleCnt.getVal(), eleCnt.getVal() - prevEleCnt);
          return prevVertCnt != vertCnt.getVal();
        };

        int cnt = 0;
        while (probeSize()) {
          pars.resize(vertCnt.getVal());
          eles.resize(eleCnt.getVal());
          cudaPol(range(prevEleCnt),
                  [eles = proxy<execspace_e::cuda>({}, eles),
                   pars = proxy<execspace_e::cuda>({}, pars),
                   vertOffsets = proxy<execspace_e::cuda>(vertOffsets),
                   eleOffsets = proxy<execspace_e::cuda>(eleOffsets), dx,
                   rho = model.density] __device__(int ei) mutable {
                    if (auto vertId = vertOffsets[ei]; vertId >= 0) {
                      /// inds, xs
                      int inds[3] = {(int)eles("inds", 0, ei),
                                     (int)eles("inds", 1, ei),
                                     (int)eles("inds", 2, ei)};
                      zs::vec<float, 3> xs[3]{pars.pack<3>("pos", inds[0]),
                                              pars.pack<3>("pos", inds[1]),
                                              pars.pack<3>("pos", inds[2])};
                      auto c = (xs[0] + xs[1] + xs[2]) / 3.f;
                      /// vel, C, F, d
                      auto vole_div_3 = eles("vol", ei) / 3.f;
                      auto vele = eles.pack<3>("vel", ei);
                      auto Ce = eles.pack<3, 3>("C", ei);
                      auto Fe = eles.pack<3, 3>("F", ei);
                      auto d2 = col(eles.pack<3, 3>("d", ei), 2);

                      auto eleId = eleOffsets[ei];

                      /// spawn new elements
                      // elem: vol (mass), pos, vel, C, F, d, Dinv,  inds
                      using mat3 = zs::vec<double, 3, 3>;
                      using vec3 = zs::vec<float, 3>;

                      /// remove this element
                      for (int i = 0; i != 3; ++i) {
                        atomic_add(exec_cuda, &pars("vol", inds[i]),
                                   -vole_div_3);
                        atomic_add(exec_cuda, &pars("mass", inds[i]),
                                   -vole_div_3 * rho);
                      }

                      /// spawn new vertex
                      pars("vol", vertId) = vole_div_3;
                      pars("mass", vertId) = vole_div_3 * rho;
                      pars.tuple<3>("pos", vertId) = c;
                      pars.tuple<3>("vel", vertId) = vele;
                      pars.tuple<9>("F", vertId) = Fe;
                      pars.tuple<9>("C", vertId) = Ce;
#if 0
                        if (pars.hasProperty("a"))
                          pars.tuple<3>("a", vertId) =
                              (pars.pack<3>("a", inds[0]),
                                   pars.pack<3>("a", inds[1]),
                                   pars.pack<3>("a", inds[2])) / 3.f;
                        if (pars.hasProperty("logJp"))
                          pars("logJp", vertId) =
                              (pars("logJp", inds[0]) + pars("logJp", inds[1]) +
                               pars("logJp", inds[2])) /
                              3.f;
#endif
              // no need to worry about the additional attibutes,
              // since they are irrelevant for the simulation
              // <a, b, c>
#define CONSTRUCT_ELEMENT(e, ia, ib, a, b)                                     \
  {                                                                            \
    const auto triArea = [](const auto &p0, const auto &p1, const auto &p2) {  \
      return (p1 - p0).cross(p2 - p0).norm() * 0.5f;                           \
    };                                                                         \
    auto vole = triArea(a, b, c) * dx;                                         \
    eles("mass", e) = vole * rho;                                              \
    eles("vol", e) = vole;                                                     \
    eles.tuple<3>("pos", e) = (a + b + c) / 3.f;                               \
    eles.tuple<3>("vel", e) = vele;                                            \
    eles.tuple<9>("C", e) = Ce;                                                \
    eles.tuple<9>("F", e) = Fe;                                                \
    mat3 d{};                                                                  \
    auto d0 = b - a;                                                           \
    d(0, 0) = d0[0];                                                           \
    d(1, 0) = d0[1];                                                           \
    d(2, 0) = d0[2];                                                           \
    auto d1 = c - a;                                                           \
    d(0, 1) = d1[0];                                                           \
    d(1, 1) = d1[1];                                                           \
    d(2, 1) = d1[2];                                                           \
    d(0, 2) = d2[0];                                                           \
    d(1, 2) = d2[1];                                                           \
    d(2, 2) = d2[2];                                                           \
    eles.tuple<9>("d", e) = d;                                                 \
    eles.tuple<9>("Dinv", e) = zs::inverse(d) * Fe;                            \
    eles("inds", 0, e) = ia;                                                   \
    eles("inds", 1, e) = ib;                                                   \
    eles("inds", 2, e) = vertId;                                               \
    atomic_add(exec_cuda, &pars("vol", ia), vole / 3.f);                       \
    atomic_add(exec_cuda, &pars("mass", ia), vole *rho / 3.f);                 \
    atomic_add(exec_cuda, &pars("vol", ib), vole / 3.f);                       \
    atomic_add(exec_cuda, &pars("mass", ib), vole *rho / 3.f);                 \
  }

                      CONSTRUCT_ELEMENT(ei, inds[0], inds[1], xs[0], xs[1]);
                      CONSTRUCT_ELEMENT(eleId, inds[1], inds[2], xs[1], xs[2]);
                      CONSTRUCT_ELEMENT((eleId + 1), inds[2], inds[0], xs[2],
                                        xs[0]);
                    }
                  });
          fmt::print("done refinement iter [{}].\n", cnt);
          if (cnt++ >= limits<int>::max())
            break;
        }
        fmt::print("finished surface mesh refinement in {} iterations.\n", cnt);
      } // surface mesh
    }

    fmt::print(fg(fmt::color::cyan), "done executing RefineMeshParticles\n");
    set_output("ZSParticles", get_input("ZSParticles"));
  }
};

ZENDEFNODE(RefineMeshParticles, {
                                    {"ZSParticles", {"float", "dx", "0.1"}},
                                    {"ZSParticles"},
                                    {},
                                    {"MPM"},
                                });

struct UpdateZSPrimitiveSequence : INode {
  void apply() override {
    using namespace zs;
    fmt::print(fg(fmt::color::green),
               "begin executing UpdateZSPrimitiveSequence\n");

    auto zsprimseq = get_input<ZenoPrimitiveSequence>("ZSPrimitiveSequence");
    auto dt = get_input2<float>("stepdt");
    if (zsprimseq->valid()) {
      auto cudaPol = cuda_exec().device(0);
      auto numV = zsprimseq->numParticles();
      auto numE = zsprimseq->numElements();
      cudaPol(
          Collapse{numV},
          [prev = proxy<execspace_e::cuda>({}, zsprimseq->getPrevParticles()),
           next = proxy<execspace_e::cuda>({}, zsprimseq->getNextParticles()),
           dt] __device__(int pi) mutable {
            prev.tuple<3>("pos", pi) =
                prev.pack<3>("pos", pi) + prev.pack<3>("vel", pi) * dt;
          });
      cudaPol(
          Collapse{numE},
          [prev = proxy<execspace_e::cuda>({}, zsprimseq->getPrevElements()),
           next = proxy<execspace_e::cuda>({}, zsprimseq->getNextElements()),
           dt] __device__(int ei) mutable {
            prev.tuple<3>("pos", ei) =
                prev.pack<3>("pos", ei) + prev.pack<3>("vel", ei) * dt;
          });
    }

    fmt::print(fg(fmt::color::cyan),
               "done executing UpdateZSPrimitiveSequence\n");
    set_output("ZSLevelSetSequence", std::move(zsprimseq));
  }
};
ZENDEFNODE(UpdateZSPrimitiveSequence,
           {
               {"ZSPrimitiveSequence", {"float", "stepdt", "0.1"}},
               {"ZSPrimitiveSequence"},
               {},
               {"MPM"},
           });

} // namespace zeno
