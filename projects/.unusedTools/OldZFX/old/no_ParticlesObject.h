#pragma once

#include <zeno/zeno.h>
#include "particles.h"

struct ParticlesObject : zeno::IObjectClone<ParticlesObject> {
    Particles pars;
};
