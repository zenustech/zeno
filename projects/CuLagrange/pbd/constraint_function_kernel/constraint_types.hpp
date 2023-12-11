#pragma once

namespace zeno { namespace PBD_CONSTRAINT {

constexpr auto CONSTRAINT_KEY = "XPBD_CONSTRAINT";
constexpr auto CONSTRAINT_TARGET = "XPBD_CONSTRAINT_TARGET";
constexpr auto CONSTRAINT_COLOR_OFFSET = "XPBD_CONSTRAINT_OFFSET";

constexpr auto NM_DCD_COLLISIONS = "NM_DCD_COLLISIONS";
constexpr auto GLOBAL_DCD_THICKNESS = "GLOBAL_DCD_THICKNESS";

constexpr auto PREVIOUS_COLLISION_TARGET = "PREVIOUS_COLLISION_TARGET";
// constexpr auto DCD_COLLISIONS_MESH_COLLIDER = "DCD_COLLISION_MESH_COLLIDER";

enum category_c : int {
    edge_length_constraint,
    isometric_bending_constraint,
    dihedral_bending_constraint,
    dihedral_spring_constraint,
    p_kp_collision_constraint,
    p_p_collision_constraint,
    vert_bending_spring,
    tri_bending_spring,
    pt_pin_constraint,
    dcd_collision_constraint,
    ccd_boundary_collision_constraint,
    volume_pin_constraint,
    bending,
    follow_animation_constraint,
    empty_constraint,
};

};
};