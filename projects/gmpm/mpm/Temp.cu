#include "../Structures.hpp"
#include "../Utils.hpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include <zeno/types/DictObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

struct AdjustParticleMaterial : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing AdjustParticleMaterial\n");
    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
    auto cudaPol = zs::cuda_exec();
    for (auto &&parObjPtr : parObjPtrs) {
      if (parObjPtr->category != ZenoParticles::mpm)
        continue;
      auto &pars = parObjPtr->getParticles();
      if (!pars.hasProperty("mu") || !pars.hasProperty("lam")) {
        std::vector<zs::PropertyTag> auxAttribs{{"mu", 1}, {"lam", 1}};
        pars.append_channels(cudaPol, auxAttribs);
      }
      auto Ehard = get_input2<float>("E_hard");
      auto Esoft = get_input2<float>("E_soft");
      auto nuhard = get_input2<float>("nu_hard");
      auto nusoft = get_input2<float>("nu_soft");
      auto endhard = get_input2<float>("end_hard");
      auto endsoft = get_input2<float>("end_soft");
      auto orient = get_param<std::string>("orientation");
      const int d = (orient == "x" ? 0 : (orient == "y" ? 1 : 2));
      //
      cudaPol(zs::range(pars.size()),
              [pars = zs::proxy<zs::execspace_e::cuda>({}, pars), Ehard, Esoft,
               nuhard, nusoft, endhard, endsoft, span = endhard - endsoft,
               d] __device__(auto pi) mutable {
                auto ratio = (endhard - pars("x", d, pi)) / span;
                ratio = __saturatef(ratio);
                auto E = Esoft * ratio + Ehard * (1.f - ratio);
                auto nu = nusoft * ratio + nuhard * (1.f - ratio);
                auto [mu, lam] = zs::lame_parameters(E, nu);
                pars("mu", pi) = mu;
                pars("lam", pi) = lam;
              });
    }

    fmt::print(fg(fmt::color::cyan), "done executing AdjustParticleMaterial\n");
    set_output("ZSParticles", get_input("ZSParticles"));
  }
};

ZENDEFNODE(AdjustParticleMaterial, {
                                       {"ZSParticles",
                                        {"float", "E_hard", "100"},
                                        {"float", "E_soft", "50"},
                                        {"float", "nu_hard", "0.4"},
                                        {"float", "nu_soft", "0.2"},
                                        {"float", "end_hard", "1"},
                                        {"float", "end_soft", "-0.5"}},
                                       {"ZSParticles"},
                                       {{"enum x y z", "orientation", "y"}},
                                       {"MPM"},
                                   });

} // namespace zeno