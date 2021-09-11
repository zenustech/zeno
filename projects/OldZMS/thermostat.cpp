#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>

using namespace zeno;

struct ScaleVelocity: zeno::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    float temperature = get_input("temperature")->as<NumericObject>()->get<float>();
    float rel_err = std::get<float>(get_param("rel_error_to_scale"));
    // random initialize for now
    auto &vel = prim->attr<zeno::vec3f>("vel");
    int n = vel.size();
    float ek = 0; // 2 * Ek
    #pragma omp parallel for reduction(+: ek)
    for (int i = 0; i < n; i++) {
        ek += zeno::lengthsq(vel[i]); // is actually sqr length
    }
    float temp_scale = sqrtf(3 * temperature * n / ek);
    rel_err = std::fmin(std::fmax(0, rel_err), 1);
    if (temp_scale > 1.0 + rel_err || temp_scale < 1.0 - rel_err) {
        printf("Temperature scaled by %f\n", temp_scale);
        #pragma omp parallel for 
        for (int i = 0; i < n; i++) {
            vel[i] *= temp_scale;
        } 
    }
    auto energy = zeno::IObject::make<NumericObject>();
    energy->set<float>(ek * 0.5);
    set_output("prim", get_input("prim"));
    set_output("kinetic energy", energy);
  }

};

static int defScaleVelocity = zeno::defNodeClass<ScaleVelocity>("ScaleVelocity",
    { /* inputs: */ {
    "prim",
    "temperature",
    }, /* outputs: */ {
    "prim",
    "kinetic energy",
    }, /* params: */ {
        {"float", "rel_error_to_scale", "0.05"},
    }, /* category: */ {
    "Molecular",
    }});
