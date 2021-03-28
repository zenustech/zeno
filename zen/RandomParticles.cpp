#include <zen/zen.h>
#include <zen/ParticlesObject.h>
#include <cstring>

namespace zenbase {

struct RandomParticles : zen::INode {
  virtual void apply() override {
    int count = std::get<int>(get_param("count"));

    auto pars = zen::IObject::make<ParticlesObject>();

    for (int i = 0; i < count; i++) {
      glm::vec3 p(drand48() * 2 - 1, drand48( * 2 - 1), drand48() * 2 - 1);
      glm::vec3 v(0);

      pars->pos.push_back(p);
      pars->vel.push_back(v);
    }

    set_output("pars", pars);
  }
};

static int defRandomParticles = zen::defNodeClass<RandomParticles>("RandomParticles",
    { /* inputs: */ {
    }, /* outputs: */ {
    "pars",
    }, /* params: */ {
    {"int", "count", ""},
    }, /* category: */ {
    "particles",
    }});

}
