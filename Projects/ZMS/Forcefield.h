#pragma once

#include <zen/zen.h>
#include <zen/vec.h>
#include "Interaction.h"
#include <vector>
#include <array>

struct ForceFieldObject : zen::Object<ForceFieldObject> {

    // We assume intermolecular potential is always pairwise
    IPairwiseInteraction *nonbond;
    // Skip intramolecular and coulomb potential for now
    // PairwiseInteraction *coulomb;
    // BondedInteraction bonded;
    IExternalInteraction *external;

    // Minimum image convention
    zen::vec3f distance(zen::vec3f ri, zen::vec3f rj, float boxlength);

    void force(std::vector<zen::vec3f, std::allocator<zen::vec3f>> &pos,
            std::vector<zen::vec3f, std::allocator<zen::vec3f>> &acc,
            float boxlength);

    float energy(std::vector<zen::vec3f, std::allocator<zen::vec3f>> &pos,
            float boxlength);
};
