#pragma once

#include <zeno/zeno.h>
#include <zeno/vec.h>
#include <functional>
#include <vector>
#include <array>

struct IPairwiseInteraction: zeno::IObject {
    float rcut = 0.0f; // 0 means no cutoff
    float rcutsq, ecut;
    // Virial (force * r) is more useful than plain force
    // Both virial and energy takes SQUARED distance as argument.
    virtual float virial(float r2) = 0;
    virtual float energy(float r2) = 0;
};

struct IExternalInteraction: zeno::IObject {
    float rcut = 0.0f; // 0 means no cutoff
    virtual float force(zeno::vec3f p) = 0;
    virtual float energy(zeno::vec3f p) = 0;
};


