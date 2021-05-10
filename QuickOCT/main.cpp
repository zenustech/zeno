#include <zen/zen.h>
#include <zen/PrimitiveObject.h>

using namespace zenbase;


struct FishYields : zen::INode {
  virtual void apply() override {
    auto stars = get_input("stars")->as<PrimitiveObject>();
    auto rate = std::get<int>(get_param("rate"));

    for (auto &[_, arr]: stars->m_attrs) {
        std::visit([rate](auto &arr) {
            for (int i = 0, j = 0; j < arr.size(); i++, j += rate) {
                arr[i] = arr[j];
            }
        }, arr);
    }
    size_t new_size = stars->size() / rate;
    printf("fish yields new_size = %zd\n", new_size);
    stars->resize(new_size);

    set_output_ref("stars", get_input_ref("stars"));
  }
};

static int defFishYields = zen::defNodeClass<FishYields>("FishYields",
    { /* inputs: */ {
    "stars",
    }, /* outputs: */ {
    "stars",
    }, /* params: */ {
    {"int", "rate", "1 1"},
    }, /* category: */ {
    "NBodySolver",
    }});


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
    const float eps = 1e-3;
    float r = eps * eps + zen::dot(rij, rij);
    return rij / (r * zen::sqrt(r));
}


struct ComputeGravity : zen::INode {
  virtual void apply() override {
    auto stars = get_input("stars")->as<PrimitiveObject>();
    auto &mass = stars->attr<float>("mass");
    auto &pos = stars->attr<zen::vec3f>("pos");
    auto &vel = stars->attr<zen::vec3f>("vel");
    auto &acc = stars->attr<zen::vec3f>("acc");
    auto G = std::get<float>(get_param("G"));
    auto eps = std::get<float>(get_param("eps"));
    printf("computing gravity...\n");
    for (int i = 0; i < stars->size(); i++) {
        acc[i] = zen::vec3f(0);
    }
    #pragma omp parallel for
    for (int i = 0; i < stars->size(); i++) {
        for (int j = i + 1; j < stars->size(); j++) {
            auto rij = pos[j] - pos[i];
            float r = eps * eps + zen::dot(rij, rij);
            rij /= r * zen::sqrt(r);
            acc[i] += mass[j] * rij;
            acc[j] -= mass[i] * rij;
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
    {"float", "eps", "0.0001 0"},
    }, /* category: */ {
    "NBodySolver",
    }});
