#include <stdexcept>
#include <zeno/ParticlesObject.h>
#include <zeno/zeno.h>

#include "../ZensimGeometry.h"
#include "../ZensimObject.h"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"

namespace zeno {

struct PairZensimParticles : zeno::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing PairZensimParticles\n");
    ZenoParticleObjects ret{};
    auto mergeParticles = [&](std::string paramStr) {
      if (!has_input(paramStr))
        return;
      if (get_input(paramStr)->as<ZenoParticles>())
        ret.push_back(get_input(paramStr)->as<ZenoParticles>());
      else if (get_input(paramStr)->as<ZenoParticleList>()) {
        ZenoParticleObjects &pobjs =
            get_input(paramStr)->as<ZenoParticleList>()->get();
        for (auto &&pobj : pobjs)
          ret.push_back(pobj);
      }
#if 0
       else
        throw std::runtime_error(
            "particle object merging failure: invalid param type");
#endif
    };
    mergeParticles("ZSParticlesA");
    mergeParticles("ZSParticlesB");

    auto particleList = zeno::IObject::make<ZenoParticleList>();
    particleList->get() = std::move(ret);
    set_output("ZensimParticleList", particleList);
    fmt::print(fg(fmt::color::cyan), "done executing PairZensimParticles\n");
  }
};

static int defPairZensimParticles = zeno::defNodeClass<PairZensimParticles>(
    "PairZensimParticles", {/* inputs: */ {"ZSParticlesA", "ZSParticlesB"},
                            /* outputs: */
                            {"ZensimParticleList"},
                            /* params: */
                            {
                                //{"float", "dx", "0.08 0"},
                            },
                            /* category: */
                            {"ZensimGeometry"}});

} // namespace zeno