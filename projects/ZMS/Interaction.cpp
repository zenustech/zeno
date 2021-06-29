#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include "Interaction.h"

using namespace zeno;

struct LennardJonesInteraction: IPairwiseInteraction {
    float sigma6, epsilon;
    // Virial (force * r) is more useful than plain force
    // Both virial and energy takes SQUARED distance as argument.
    virtual float virial(float r2) {
      float r6 = r2 * r2 * r2;
      float v = 24 * epsilon * (2 * sigma6 * sigma6 / (r6 * r6) - sigma6 / r6);
      //printf("%f %f %f %f\n", r2, sigma6, epsilon, v);
      return v;
    }

    virtual float energy(float r2) {
      float r6 = r2 * r2 * r2;
      return 4 * epsilon * (sigma6 * sigma6 / (r6 * r6) - sigma6 / r6) - ecut;
    }

};

struct LennardJones: zeno::INode {
  virtual void apply() override {
    auto rcut = get_input("rcut")->as<NumericObject>()->get<float>();
    auto sigma = get_input("sigma")->as<NumericObject>()->get<float>();
    auto epsilon = get_input("epsilon")->as<NumericObject>()->get<float>();
    auto sigma3 = sigma * sigma * sigma;
    
    auto lj = zeno::IObject::make<LennardJonesInteraction>();
    lj->rcut = rcut;
    lj->rcutsq = rcut * rcut;
    lj->sigma6 = sigma3 * sigma3;
    lj->epsilon = epsilon;
    lj->ecut = lj->energy(rcut * rcut);
    printf("LJ: %f %f %f %f\n", sigma, epsilon, lj->rcut, lj->ecut);
    set_output("lennard-jones", lj);
  }
};

static int defLennardJones = zeno::defNodeClass<LennardJones>("LennardJones",
    { /* inputs: */ {
    "rcut",
    "sigma",
    "epsilon",
    }, /* outputs: */ {
    "lennard-jones",
    }, /* params: */ {
    }, /* category: */ {
    "Molecular",
    }});
