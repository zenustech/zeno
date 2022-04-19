#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include "Forcefield.h"

using namespace zeno;

// Minimum image convention
zeno::vec3f ForceFieldObject::distance(zeno::vec3f ri, zeno::vec3f rj, float boxlength) {
    auto d = ri - rj;
    for (int i = 0; i < 3; i++) {
        while (d[i] <= -0.5 * boxlength) d[i] += boxlength;
        while (d[i] > 0.5 * boxlength) d[i] -= boxlength;
    }
    return d;
}

void ForceFieldObject::force(std::vector<zeno::vec3f, std::allocator<zeno::vec3f>> &pos,
        std::vector<zeno::vec3f, std::allocator<zeno::vec3f>> &acc,
        float boxlength) {
    float mass = 1.0f; // TODO: add primitive type attribute and mass table
    // External potental
    int n = pos.size();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        acc[i] = zeno::vec3f(0.0f);
        if (external != nullptr) {
            acc[i] += external->force(pos[i]) / mass;
        }
    }
    // Pairwise potential with cutoff
    if (nonbond != nullptr) {
        # pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    auto d = distance(pos[i], pos[j], boxlength);
                    float r2 = zeno::lengthSquared(d);
                    if (r2 <= nonbond->rcutsq) {
                        //printf("%d %d %f %f %f %f\n", i, j, d[0], d[1], d[2], r2);
                        auto force = nonbond->virial(r2) * d / r2;
                        // Pairwise force is always along the distance direction
                        acc[i] += force;
                    }
                }
            }
        }
    }
}

float ForceFieldObject::energy(std::vector<zeno::vec3f, std::allocator<zeno::vec3f>> &pos,
        float boxlength) {
    float ep = 0.0f;
    // External potental
    int n = pos.size();
    if (external != nullptr) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {  
            ep += external->energy(pos[i]);
        }
    }
    // Pairwise potential with cutoff
    if (nonbond != nullptr) {
        # pragma omp parallel for reduction(+: ep)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                auto d = distance(pos[i], pos[j], boxlength);
                float r2 = zeno::lengthSquared(d);
                //printf("%f %f %f %f\n", r2, nonbond->rcut, nonbond->rcutsq, nonbond->ecut);
                if (r2 <= nonbond->rcutsq)  {
                    ep += nonbond->energy(r2);
                }
            }
        }
    }
    return ep;
}


struct ApplyForce: zeno::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto &pos = prim->attr<zeno::vec3f>("pos");
    auto &acc = prim->attr<zeno::vec3f>("acc");
    auto boxlength = get_input("boxlength")->as<NumericObject>()->get<float>();
    
    auto forcefield = get_input("forcefield")->as<ForceFieldObject>();
    
    // We strictly use conservative forces (no velocity argument)
    // Mass of particles is contained in the force field;
    // TODO: add primitive type attribute
    forcefield->force(pos, acc, boxlength);
    printf("Apply force\n");
    set_output("prim", get_input("prim"));
  }
};

static int defApplyForce = zeno::defNodeClass<ApplyForce>("ApplyForce",
    { /* inputs: */ {
    "prim",
    "forcefield",
    "boxlength",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "Molecular",
    }});


struct ForceField: zeno::INode {
  virtual void apply() override {
    auto nonbond = get_input("nonbond");
    //auto external = get_input("external");
    auto forcefield = zeno::IObject::make<ForceFieldObject>();
    if (nonbond != nullptr) {
      forcefield->nonbond = nonbond->as<IPairwiseInteraction>();
    }
    //if (external != nullptr) {
    //  forcefield->external = external->as<ExternalInteraction>();
    //}
    set_output("forcefield", forcefield);
  }
};

static int defForceField = zeno::defNodeClass<ForceField>("ForceField",
    { /* inputs: */ {
    "nonbond",
    "coulomb",
    "bonded",
    "external",
    "mass",
    }, /* outputs: */ {
    "forcefield",
    }, /* params: */ {
    }, /* category: */ {
    "Molecular",
    }});
