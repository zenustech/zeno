#pragma once

#include <zeno/zen.h>
#include <zeno/vec.h>
#include <functional>
#include <vector>
#include <array>

struct IPairwiseInteraction: zen::IObject {
    float rcut = 0.0f; // 0 means no cutoff
    float rcutsq, ecut;
    // Virial (force * r) is more useful than plain force
    // Both virial and energy takes SQUARED distance as argument.
    virtual float virial(float r2) = 0;
    virtual float energy(float r2) = 0;
};

struct IExternalInteraction: zen::IObject {
    float rcut = 0.0f; // 0 means no cutoff
    virtual float force(zen::vec3f p) = 0;
    virtual float energy(zen::vec3f p) = 0;
};


