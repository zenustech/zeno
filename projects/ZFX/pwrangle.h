#pragma once

#include "program.h"
#include "particles.h"

void vectors_wrangle(Program const *prog,
    std::vector<std::vector<float> *> const &arrs);
void particles_wrangle(Program const *prog, Particles const *pars);
