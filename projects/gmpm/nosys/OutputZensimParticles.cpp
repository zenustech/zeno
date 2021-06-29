#include <zeno/ParticlesObject.h>
#include <zeno/zen.h>

#include "../ZensimGeometry.h"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/simulation/mpm/Simulator.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"

namespace zen {

struct OutputZensimParticles : zen::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing OutputZensimParticles\n");
    auto inParticles = get_input("ZensimParticles")->as<ZenoParticles>();
    auto path = std::get<std::string>(get_param("path"));
    zs::match([&path](auto &p) {
      zs::write_partio<float, 3>(path, p.retrievePositions());
    })(inParticles->get());
    fmt::print(fg(fmt::color::cyan), "done executing OutputZensimParticles\n");
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

} // namespace zen