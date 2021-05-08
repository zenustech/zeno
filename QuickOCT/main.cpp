#include <zen/zen.h>
#include <zen/PrimitiveObject.h>

using namespace zenbase;


struct AdvectStars : zen::INode {
  virtual void apply() override {
    auto stars = get_input("stars")->as<PrimitiveObject>();
    auto &pos = stars->attr<glm::vec3>("pos");
    auto &vel = stars->attr<glm::vec3>("vel");
    auto &acc = stars->attr<glm::vec3>("acc");
    auto dt = std::get<float>(get_param("dt"));
    #pragma omp parallel for
    for (int i = 0; i < stars->size(); i++) {
        pos[i] += vel[i] * dt + acc[i] * (dt * dt / 2);
        vel[i] += acc[i] * dt;
    }
  }
};

static int defAdvectStars = zen::defNodeClass<AdvectStars>("AdvectStars",
    { /* inputs: */ {
    "stars",
    }, /* outputs: */ {
    }, /* params: */ {
    {"float", "dt", "0.01 0"},
    }, /* category: */ {
    "NBodySolver",
    }});
