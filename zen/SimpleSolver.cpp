#include <zen/zen.h>
#include <zen/ParticlesObject.h>
#include <omp.h>

namespace zenbase {

struct SimpleSolver : zen::INode {
  ParticlesObject *pars;
  float dt{0};
  glm::vec3 G{0, 0, 0};

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
#pragma omp parallel for
    for (int i = 0; i < pars->size(); i++) {
      pars->pos[i] += pars->vel[i] * dt;
      pars->vel[i] += G * dt;
    }
  }

  void init() {
    // initializing members
    pars = new_member<ParticlesObject>("pars");
    // initializing parameters
    dt = std::get<float>(get_param("dt"));
    G = zen::get_float3<glm::vec3>(get_param("G"));
    // initializing internal data
    auto ini_pars = get_input("ini_pars")->as<ParticlesObject>();
    *pars = *ini_pars;  // deep-copy
  }
};


static int defSimpleSolver = zen::defNodeClass<SimpleSolver>("SimpleSolver",
    { /* inputs: */ {
    "ini_pars",
    }, /* outputs: */ {
    "pars",
    }, /* params: */ {
    {"float", "dt", "0.04"},
    {"float3", "G", "0 0 1"},
    }, /* category: */ {
    "solver",
    }});

}
