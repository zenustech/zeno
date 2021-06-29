#include <zeno/ParticlesObject.h>
#include <zeno/zeno.h>

#include "../ZensimGeometry.h"
#include "../ZensimModel.h"
#include "zensim/simulation/mpm/Simulator.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"

namespace zen {

struct ToZensimParticles : zen::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ToZensimParticles\n");
    auto &model = get_input("ZSModel")->as<ZenoConstitutiveModel>()->get();
    auto inParticles = get_input("ParticleObject")->as<ParticlesObject>();
    auto outParticles = zen::IObject::make<ZenoParticles>();
    const auto size = inParticles->size();
    outParticles->get() = zs::Particles<zs::f32, 3>{};

    const bool hasPlasticity =
        model.index() ==
            magic_enum::enum_integer(zs::constitutive_model_e::DruckerPrager) ||
        model.index() ==
            magic_enum::enum_integer(zs::constitutive_model_e::NACC);
    const bool hasF =
        model.index() !=
        magic_enum::enum_integer(zs::constitutive_model_e::EquationOfState);
    {
      zs::match(
          [&inParticles, size, hasF, hasPlasticity](auto &model, auto &res) {
            using PT = zs::remove_cvref_t<decltype(res)>;
            using T = typename PT::T;
            using TV = typename PT::TV;
            using TM = typename PT::TM;
            using ModelT = zs::remove_cvref_t<decltype(model)>;

            res.M = zs::Vector<T>{size, zs::memsrc_e::um, 0};
            res.X = zs::Vector<TV>{size, zs::memsrc_e::um, 0};
            res.V = zs::Vector<TV>{size, zs::memsrc_e::um, 0};
            if (hasF)
              res.F = zs::Vector<TM>{size, zs::memsrc_e::um, 0};
            else
              res.J = zs::Vector<T>{size, zs::memsrc_e::um, 0};
            res.C = zs::Vector<TM>{size, zs::memsrc_e::um, 0};
            if (hasPlasticity)
              res.logJp = zs::Vector<T>{size, zs::memsrc_e::um, 0};

            auto mass = model.volume * model.rho;
            for (auto &&m : res.M)
              m = mass;
            for (auto &&[dst, src] : zs::zip(res.X, inParticles->pos))
              dst = TV{src[0], src[1], src[2]};
            for (auto &&[dst, src] : zs::zip(res.V, inParticles->vel))
              dst = TV{src[0], src[1], src[2]};
            for (auto &&c : res.C)
              c = TM{0, 0, 0, 0, 0, 0, 0, 0, 0};
            if (hasF)
              for (auto &&f : res.F)
                f = TM{1, 0, 0, 0, 1, 0, 0, 0, 1};
            else
              for (auto &&j : res.J)
                j = 1.f;
            if constexpr (std::is_same_v<ModelT, zs::NACCConfig> ||
                          std::is_same_v<ModelT, zs::DruckerPragerConfig>) {
              for (auto &&logjp : res.logJp)
                logjp = model.logJp0;
            }
          })(model, outParticles->get());
    }
    fmt::print(fg(fmt::color::cyan), "done executing ToZensimParticles\n");
    set_output("ZensimParticles", outParticles);
  }
};

static int defToZensimParticles = zen::defNodeClass<ToZensimParticles>(
    "ToZensimParticles", {/* inputs: */ {"ZSModel", "ParticleObject"},
                          /* outputs: */
                          {"ZensimParticles"},
                          /* params: */
                          {
                              //{"float", "dx", "0.08 0"},
                          },
                          /* category: */
                          {"ZensimGeometry"}});

} // namespace zen