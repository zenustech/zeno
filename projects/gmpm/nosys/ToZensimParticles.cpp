#include <zeno/ParticlesObject.h>
#include <zeno/zeno.h>

#include "../ZensimGeometry.h"
#include "../ZensimModel.h"
#include "zensim/container/Vector.hpp"
#include "zensim/simulation/mpm/Simulator.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"

namespace zeno {

struct ToZensimParticles : zeno::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ToZensimParticles\n");
    auto &model = get_input("ZSModel")->as<ZenoConstitutiveModel>()->get();
    auto inParticles = get_input("ParticleObject")->as<ParticlesObject>();
    auto outParticles = zeno::IObject::make<ZenoParticles>();
    const auto size = inParticles->size();
    outParticles->get() = zs::Particles<zs::f32, 3>{size, zs::memsrc_e::um, 0};

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

            res.addAttr("mass", zs::scalar_v);
            res.addAttr("vel", zs::vector_v);
            if (hasF)
              res.addAttr("F", zs::matrix_v);
            else
              res.addAttr("J", zs::scalar_v);
            res.addAttr("C", zs::matrix_v);
            if (hasPlasticity)
              res.addAttr("logJp", zs::scalar_v);

            auto mass = model.volume * model.rho;
            for (auto &&m : res.attrScalar("mass"))
              m = mass;
            for (auto &&[dst, src] :
                 zs::zip(res.attrVector("pos"), inParticles->pos)) {
              if constexpr (TV::extent == 3)
                dst = TV{src[0], src[1], src[2]};
              else
                throw std::runtime_error("not implemented");
            }
            for (auto &&[dst, src] :
                 zs::zip(res.attrVector("vel"), inParticles->vel)) {
              if constexpr (TV::extent == 3)
                dst = TV{src[0], src[1], src[2]};
              else
                throw std::runtime_error("not implemented");
            }
            for (auto &&c : res.attrMatrix("C"))
              c = TM::zeros();
            if (hasF)
              for (auto &&f : res.attrMatrix("F")) {
                if constexpr (TV::extent == 3) {
                  f = TM{1, 0, 0, 0, 1, 0, 0, 0, 1};
                } else
                  throw std::runtime_error("not implemented");
              }
            else
              for (auto &&j : res.attrScalar("J"))
                j = 1.f;
            if constexpr (std::is_same_v<ModelT, zs::NACCConfig> ||
                          std::is_same_v<ModelT, zs::DruckerPragerConfig>) {
              for (auto &&logjp : res.attrScalar("logJp"))
                logjp = model.logJp0;
            }
          })(model, outParticles->get());
    }
    outParticles->model = model;
    fmt::print(fg(fmt::color::cyan), "done executing ToZensimParticles\n");
    set_output("ZensimParticles", outParticles);
  }
};

static int defToZensimParticles = zeno::defNodeClass<ToZensimParticles>(
    "ToZensimParticles", {/* inputs: */ {"ZSModel", "ParticleObject"},
                          /* outputs: */
                          {"ZensimParticles"},
                          /* params: */
                          {
                              //{"float", "dx", "0.08 0"},
                          },
                          /* category: */
                          {"ZensimGeometry"}});

} // namespace zeno