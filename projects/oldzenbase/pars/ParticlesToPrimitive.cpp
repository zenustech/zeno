#include <zeno/zeno.h>
#include <zeno/ParticlesObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {

struct ParticlesToPrimitive : zeno::INode{
    virtual void apply() override {
    auto pars = get_input("pars")->as<ParticlesObject>();
    auto result = zeno::IObject::make<PrimitiveObject>();
    result->add_attr<zeno::vec3f>("pos");
    result->add_attr<zeno::vec3f>("vel");
    result->resize(std::max(pars->pos.size(), pars->vel.size()));

#pragma omp parallel for
    for(int i=0;i<pars->pos.size();i++) {
        result->attr<zeno::vec3f>("pos")[i] = zeno::vec3f(pars->pos[i].x,
            pars->pos[i].y, pars->pos[i].z);
    }
#pragma omp parallel for
    for(int i=0;i<pars->vel.size();i++) {
        result->attr<zeno::vec3f>("vel")[i] = zeno::vec3f(pars->vel[i].x,
            pars->vel[i].y, pars->vel[i].z);
    }
    set_output("prim", result);
  }
};

static int defParticlesToPrimitive = zeno::defNodeClass<ParticlesToPrimitive>("ParticlesToPrimitive",
    { /* inputs: */ {
        "pars",
    }, /* outputs: */ {
        "prim",
    }, /* params: */ { 
    }, /* category: */ {
    "primitive",
    }});

}
