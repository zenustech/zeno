#include <zeno/zeno.h>
#include "program.h"
#include "no_ProgramObject.h"
#include "no_ParticlesObject.h"
#include "pwrangle.h"
#include "particles.h"

using namespace zeno;

struct ParticlesWrangle : INode {
    virtual void apply() override {
        auto particles = get_input<ParticlesObject>("particles");
        auto program = get_input<ProgramObject>("program");
        particles_wrangle(&program->prog, &particles->pars);
    }
};

ZENDEFNODE(ParticlesWrangle, {
    {"particles", "program"},
    {"particles"},
    {},
    {"zenofx"},
});
