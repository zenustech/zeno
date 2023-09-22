#pragma once

namespace zeno { namespace PBD_CONSTRAINT {

constexpr auto CONSTRAINT_KEY = "XPBD_CONSTRAINT";

enum category_c : int {
    edge_length_constraint,
    isometric_bending_constraint,
    dihedral_bending_constraint,
    p_kp_collision_constraint,
    p_p_collision_constraint,
    vert_bending_spring,
    tri_bending_spring,
    bending
};

};
};