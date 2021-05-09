#include <zen/ParticlesObject.h>
#include <zen/zen.h>

#include "../ZensimGeometry.h"
#include "zensim/simulation/mpm/Simulator.hpp"

namespace zenbase {

struct ToZensimParticles : zen::INode {
  void apply() override {
    auto inParticles = get_input("ParticleObject")->as<ParticlesObject>();
    auto outParticles = zen::IObject::make<ZenoParticles>();
    const auto size = inParticles->size();
    outParticles->get() = zs::Particles<zs::f32, 3>{};
    {
      zs::match([&inParticles, size](auto &res) {
        using PT = zs::remove_cvref_t<decltype(res)>;
        using TV = typename PT::TV;
        using TM = typename PT::TM;
        res.X = zs::Vector<TV>{size, zs::memsrc_e::um, 0};
        res.V = zs::Vector<TV>{size, zs::memsrc_e::um, 0};
        for (auto &&[dst, src] : zs::zip(res.X, inParticles->pos))
          dst = TV{src[0], src[1], src[2]};
        for (auto &&[dst, src] : zs::zip(res.V, inParticles->vel))
          dst = TV{src[0], src[1], src[2]};
        res.C = zs::Vector<TM>{size, zs::memsrc_e::um, 0};
        for (auto &&c : res.C)
          c = TM{1, 0, 0, 0, 1, 0, 0, 0, 1};
      })(outParticles->get());
    }
    set_output("ZensimParticles", outParticles);
  }
};

static int defToZensimParticles = zen::defNodeClass<ToZensimParticles>(
    "ToZensimParticles", {/* inputs: */ {"ParticleObject"},
                          /* outputs: */
                          {"ZensimParticles"},
                          /* params: */
                          {
                              //{"float", "dx", "0.08 0"},
                          },
                          /* category: */
                          {"ZensimGeometry"}});

} // namespace zenbase