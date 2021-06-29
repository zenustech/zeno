#pragma once

#include <zeno/zeno.h>
#include <zeno/vec.h>
#include "Interaction.h"
#include <vector>
#include <array>

struct ForceFieldObject : zeno::IObject {

    // We assume intermolecular potential is always pairwise
    IPairwiseInteraction *nonbond;
    // Skip intramolecular and coulomb potential for now
    // PairwiseInteraction *coulomb;
    // BondedInteraction bonded;
    IExternalInteraction *external;

    // Minimum image convention
    zeno::vec3f distance(zeno::vec3f ri, zeno::vec3f rj, float boxlength);

    void force(std::vector<zeno::vec3f, std::allocator<zeno::vec3f>> &pos,
            std::vector<zeno::vec3f, std::allocator<zeno::vec3f>> &acc,
            float boxlength);

    float energy(std::vector<zeno::vec3f, std::allocator<zeno::vec3f>> &pos,
            float boxlength);
};
