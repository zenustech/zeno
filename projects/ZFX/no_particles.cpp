#include <zeno/zeno.h>
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
