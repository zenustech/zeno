#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include "no_ProgramObject.h"
#include "no_ParticlesObject.h"
#include "particles.h"

using namespace zeno;

struct MakeParticles : INode {
    virtual void apply() override {
        auto particles = std::make_shared<ParticlesObject>();
        auto nchannels = get_param<int>("nchannels");
        auto size = get_param<int>("size");
        particles->pars.set_nchannels(nchannels);
        particles->pars.resize(size);
        set_output("particles", std::move(particles));
    }
};

ZENDEFNODE(MakeParticles, {
    {},
    {"particles"},
    {{"int", "nchannels", "0"}, {"int", "size", "0"}},
    {"zenofx"},
});


struct ParticlesToPrimitive : INode {
    virtual void apply() override {
        auto prim = std::make_shared<PrimitiveObject>();
        auto pars = &get_input<ParticlesObject>("particles")->pars;
        prim->resize(pars->size());
        auto &prim_pos = prim->add_attr<vec3f>("pos");
        for (size_t i = 0; i < pars->size(); i++) {
            auto x = pars->channel(0)->at(i);
            auto y = pars->channel(1)->at(i);
            auto z = pars->channel(2)->at(i);
            prim_pos[i] = vec3f(x, y, z);
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ParticlesToPrimitive, {
    {"particles"},
    {"prim"},
    {},
    {"zenofx"},
});
