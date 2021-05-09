#include <zen/zen.h>
#include <zen/PrimitiveObject.h>

using namespace zenbase;


struct AdvectStars : zen::INode {
  virtual void apply() override {
    auto stars = get_input("stars")->as<PrimitiveObject>();
    auto &mass = stars->attr<float>("mass");
    auto &pos = stars->attr<zen::vec3f>("pos");
    auto &vel = stars->attr<zen::vec3f>("vel");
    auto &acc = stars->attr<zen::vec3f>("acc");
    auto dt = std::get<float>(get_param("dt"));
    #pragma omp parallel for
    for (int i = 0; i < stars->size(); i++) {
        pos[i] += vel[i] * dt + acc[i] * (dt * dt / 2);
        vel[i] += acc[i] * dt;
    }

    set_output_ref("stars", get_input_ref("stars"));
  }
};

static int defAdvectStars = zen::defNodeClass<AdvectStars>("AdvectStars",
    { /* inputs: */ {
    "stars",
    }, /* outputs: */ {
    "stars",
    }, /* params: */ {
    {"float", "dt", "0.01 0"},
    }, /* category: */ {
    "NBodySolver",
    }});


static zen::vec3f gfunc(zen::vec3f const &rij) {
    const float eps = 1e-4;
    float r = eps * eps + zen::dot(rij, rij);
    return rij / zen::pow(r, 3);
}


struct ComputeGravity : zen::INode {
  virtual void apply() override {
    auto stars = get_input("stars")->as<PrimitiveObject>();
    auto &mass = stars->attr<float>("mass");
    auto &pos = stars->attr<zen::vec3f>("pos");
    auto &vel = stars->attr<zen::vec3f>("vel");
    auto &acc = stars->attr<zen::vec3f>("acc");
    auto G = std::get<float>(get_param("G"));
    printf("computing gravity...\n");
    #pragma omp parallel for
    for (int i = 0; i < stars->size(); i++) {
        acc[i] = zen::vec3f(0);
        for (int j = i + 1; j < stars->size(); j++) {
            acc[i] += mass[j] * gfunc(pos[j] - pos[i]);
        }
    }
    printf("computing gravity done\n");
    for (int i = 0; i < stars->size(); i++) {
        acc[i] *= G;
    }

    set_output_ref("stars", get_input_ref("stars"));
  }
};

static int defComputeGravity = zen::defNodeClass<ComputeGravity>("ComputeGravity",
    { /* inputs: */ {
    "stars",
    }, /* outputs: */ {
    "stars",
    }, /* params: */ {
    {"float", "G", "1.0 0"},
    }, /* category: */ {
    "NBodySolver",
    }});
