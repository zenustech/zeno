#include <zen/zen.h>
#include <zen/ParticlesObject.h>
#include <zen/NumericObject.h>
#include <omp.h>

namespace zenbase {

struct NBodySolver : zen::INode {
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
    dt = get_input("dt")->as<NumericObject>()->value;
#pragma omp parallel for
    for (int i = 0; i < pars->size(); i++) {
      pars->pos[i] += pars->vel[i] * dt;
      pars->vel[i] += G * dt;
    }
  }

  void init() {
    G = zen::get_float3<glm::vec3>(get_param("G"));
    auto ini_pars = get_input("ini_pars")->as<ParticlesObject>();
    pars = new_member<ParticlesObject>("pars");
    *pars = *ini_pars;  // deep-copy
  }
};


static int defNBodySolver = zen::defNodeClass<NBodySolver>("NBodySolver",
    { /* inputs: */ {
    "ini_pars",
    "dt",
    }, /* outputs: */ {
    "pars",
    }, /* params: */ {
    {"float3", "G", "0 0 1"},
    }, /* category: */ {
    "particles",
    }});

}
