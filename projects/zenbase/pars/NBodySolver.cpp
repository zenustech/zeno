#include <zeno/zeno.h>
#include <zeno/ParticlesObject.h>
#include <zeno/NumericObject.h>
#include <glm/glm.hpp>
#include <cmath>
#include <omp.h>

namespace zeno {

struct NBodySolver : zeno::INode {
  ParticlesObject *pars;

  bool initialized{false};

  virtual void apply() override {
    // solver nodes will do init() on their first apply()
    if (!initialized) {
      initialized = true;
      init();
    } else {
      step();
    }
  }

  void step() {
    auto dt = get_input("dt")->as<NumericObject>()->get<float>();
    auto G = std::get<float>(get_param("G"));
    auto M = std::get<float>(get_param("M"));
    auto r0 = std::get<float>(get_param("r0"));

#pragma omp parallel for
    for (int i = 0; i < pars->size(); i++) {

      glm::vec3 acc(0);
      auto p = pars->pos[i];
      for (int j = 0; j < pars->size(); j++) {
        if (j == i) continue;
        auto r = p - pars->pos[j];
        auto x = std::sqrt(glm::dot(r, r));
        auto fac = G * r0 * r0 * r0 / (x * x * x + 1e-3);// + M * (std::pow(x, 13) - std::pow(x, 7));
        acc += glm::vec3(fac) * r;
      }
      pars->vel[i] += acc * dt;
    }

#pragma omp parallel for
    for (int i = 0; i < pars->size(); i++) {
      pars->pos[i] += pars->vel[i] * dt;
    }
  }

  void init() {
    auto ini_pars = get_input("ini_pars")->as<ParticlesObject>();
    pars = new_member<ParticlesObject>("pars");
    *pars = *ini_pars;  // deep-copy
  }
};


static int defNBodySolver = zeno::defNodeClass<NBodySolver>("NBodySolver",
    { /* inputs: */ {
    "ini_pars",
    "dt",
    }, /* outputs: */ {
    "pars",
    }, /* params: */ {
    {"float", "r0", "0.02"},
    {"float", "G", "-10.0"},
    {"float", "M", "0.001"},
    }, /* category: */ {
    "particles",
    }});

}
