#pragma once

namespace zeno { namespace PBD_CONSTRAINT {

constexpr auto CONSTRAINT_KEY = "XPBD_CONSTRAINT";
constexpr auto CONSTRAINT_TARGET = "XPBD_CONSTRAINT_TARGET";
constexpr auto CONSTRAINT_COLOR_OFFSET = "XPBD_CONSTRAINT_OFFSET";

constexpr auto NM_DCD_COLLISIONS = "NM_DCD_COLLISIONS";
constexpr auto GLOBAL_DCD_THICKNESS = "GLOBAL_DCD_THICKNESS";
constexpr auto ENABLE_SLIDING = "ENABLE_SLIDING";

constexpr auto ENABLE_DCD_REPULSION_FORCE = "ENABLE_DCD_REPULSION_FORCE";

constexpr auto PREVIOUS_COLLISION_TARGET = "PREVIOUS_COLLISION_TARGET";

constexpr auto TARGET_CELL_BUFFER = "TARGET_CELL_BUFFER";

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
    vertex_pin_to_cell_constraint,
    dcd_collision_constraint,
    kinematic_dcd_collision_constraint,
    self_dcd_collision_constraint,
    ccd_boundary_collision_constraint,
    volume_pin_constraint,
    bending,
    follow_animation_constraint,
    empty_constraint,
};

};
};