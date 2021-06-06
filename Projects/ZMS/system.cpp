#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/NumericObject.h>

using namespace zenbase;

struct PeriodicBoundary : zen::INode {
  virtual void apply() override {
    auto prims = get_input("prims")->as<PrimitiveObject>();
    auto &pos = prims->attr<zen::vec3f>("pos");
    float boxlength = get_input("boxlength")->as<NumericObject>()->get<float>();
    #pragma omp parallel for
    for (int i = 0; i < prims->size(); i++) {
        for (int d = 0; d < 3; d++) {
            if (pos[i][d] >= boxlength) pos[i][d] -= boxlength;
            if (pos[i][d] < 0) pos[i][d] += boxlength;
        }
    }

    set_output_ref("prims", get_input_ref("prims"));
  }
};

static int defPeriodicBoundary = zen::defNodeClass<PeriodicBoundary>("PeriodicBoundary",
    { /* inputs: */ {
    "prims",
    "boxlength",
    }, /* outputs: */ {
    "prims",
    }, /* params: */ {
    }, /* category: */ {
    "Molecular",
    }});


struct SimulationBox : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto boxlength = std::get<float>(get_param("boxlength"));
    auto n_particles = std::get<int>(get_param("n_particles"));
    prim->resize(n_particles);
    prim->add_attr<zen::vec3f>("pos");
    prim->add_attr<zen::vec3f>("vel");
    prim->add_attr<zen::vec3f>("acc");
    prim->add_attr<zen::vec3f>("clr");
    prim->add_attr<float>("mass");
    auto boxlength_obj = zen::IObject::make<NumericObject>();
    boxlength_obj->set(boxlength);
    auto n_particles_obj = zen::IObject::make<NumericObject>();
    n_particles_obj->set(n_particles);

    set_output_ref("prim", get_input_ref("prim"));
    set_output("boxlength", boxlength_obj);
    set_output("n_particles", n_particles_obj);
    
  }
};


static int defSimulationBox = zen::defNodeClass<SimulationBox>("SimulationBox",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    "boxlength",
    "n_particles",
    }, /* params: */ {
      {"float", "boxlength", "1"},
      {"int", "n_particles", "128"},
    }, /* category: */ {
    "Molecular",
    }});

struct InitializeSystem: zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto boxlength_obj = get_input("boxlength")->as<NumericObject>();
    float boxlength = boxlength_obj->get<float>();
    // random initialize for now
    auto &pos = prim->attr<zen::vec3f>("pos");
    auto &vel = prim->attr<zen::vec3f>("vel");
    auto &clr = prim->attr<zen::vec3f>("clr");
    auto &mass = prim->attr<float>("mass");

    auto n = pos.size();
    int nx = ceilf(cbrtf((float) n));
    float spacing = boxlength / nx;
    zen::vec3f vcm(0);
    for (int i = 0; i < n; i++) {
        zen::vec3i base(i / (nx * nx), (i % (nx * nx)) / nx, i % nx);
        zen::vec3f p(drand48(), drand48(), drand48());
        zen::vec3f v(drand48(), drand48(), drand48());
        pos[i] = spacing * (p * 0.4 + 0.1 + base);
        vel[i] = v;
        vcm += vel[i];
        mass[i] = 1.0f; // ad-hoc, reserved for mix-atom simulation
        clr[i] = zen::vec3f(1);
    }
    vcm /= n;
    for (int i = 0; i < n; i++) {
        vel[i] -= vcm;
    }
    set_output_ref("prim", get_input_ref("prim"));
  }

};

static int defInitializeSystem = zen::defNodeClass<InitializeSystem>("InitializeSystem",
    { /* inputs: */ {
    "prim",
    "boxlength",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "Molecular",
    }});
