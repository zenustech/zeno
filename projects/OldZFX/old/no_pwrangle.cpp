#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include "no_ParticlesObject.h"
#include "particles.h"
#include "pwrangle.h"
#include "compile.h"

using namespace zeno;

struct ParticlesWrangle : INode {
    virtual void apply() override {
        auto particles = get_input<ParticlesObject>("particles");
        auto code = get_input<StringObject>("ZFXCode")->get();
        auto prog = compile_program(code);
        particles_wrangle(prog, &particles->pars);
        set_output("particles", std::move(particles));
    }
};

ZENDEFNODE(ParticlesWrangle, {
    {"particles", "ZFXCode"},
    {"particles"},
    {},
    {"zenofx"},
});
