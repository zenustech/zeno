#include <zeno/ParticlesObject.h>
#include <zeno/zeno.h>

#include "../ZensimGeometry.h"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/simulation/mpm/Simulator.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"

namespace zeno {

struct OutputZensimParticles : zeno::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing OutputZensimParticles\n");
    auto inParticles = get_input("ZensimParticles")->as<ZenoParticles>();
    auto path = std::get<std::string>(get_param("path"));
    zs::match([&path](auto &p) {
      const auto &pos = p.retrievePositions();
      zs::write_partio<float, std::tuple_size_v<RM_CVREF_T(pos[0])>>(path, pos);
    })(inParticles->get());
    fmt::print(fg(fmt::color::cyan), "done executing OutputZensimParticles\n");
  }
};

static int defOutputZensimParticles = zeno::defNodeClass<OutputZensimParticles>(
    "OutputZensimParticles", {/* inputs: */ {"ZensimParticles"},
                              /* outputs: */
                              {},
                              /* params: */
                              {{"string", "path", ""}},
                              /* category: */
                              {"ZensimGeometry"}});

} // namespace zeno