#include <zen/ParticlesObject.h>
#include <zen/zen.h>

#include "../ZensimGeometry.h"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/simulation/mpm/Simulator.hpp"

namespace zenbase {

struct OutputZensimParticles : zen::INode {
  void apply() override {
    auto inParticles = get_input("ZensimParticles")->as<ZenoParticles>();
    auto path = std::get<std::string>(get_param("path"));
    zs::match([&path](auto &p) {
      zs::write_partio<float, 3>(path, p.retrievePositions());
    })(inParticles->get());
  }
};

static int defOutputZensimParticles = zen::defNodeClass<OutputZensimParticles>(
    "OutputZensimParticles", {/* inputs: */ {"ZensimParticles"},
                              /* outputs: */
                              {},
                              /* params: */
                              {{"string", "path", ""}},
                              /* category: */
                              {"ZensimGeometry"}});

} // namespace zenbase