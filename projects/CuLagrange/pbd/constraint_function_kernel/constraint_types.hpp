#pragma once

namespace zeno { namespace PBD_CONSTRAINT {

constexpr auto CONSTRAINT_KEY = "XPBD_CONSTRAINT";
constexpr auto CONSTRAINT_TARGET = "XPBD_CONSTRAINT_TARGET";
constexpr auto CONSTRAINT_COLOR_OFFSET = "XPBD_CONSTRAINT_OFFSET";
constexpr auto CONSTRAINT_UPDATE_INDEX_BUFFER = "CONSTRAINT_UPDATE_INDEX_BUFFER";
constexpr auto CONSTRAINT_UPDATE_OFFSETS = "CONSTRAINT_UPDATE_OFFSETS";
constexpr auto CONSTRAINT_UPDATE_BUFFER = "CONSTRAINT_Update_BUFFER";


constexpr auto NM_DCD_COLLISIONS = "NM_DCD_COLLISIONS";
constexpr auto DCD_COUNTER_BUFFER = "DCD_COUNTER_BUFFER";
constexpr auto COLLISION_BUFFER = "COLLLISION_BUFFER";
constexpr auto COLLISION_CSEE_SET = "COLLISION_CSEE_SET";
constexpr auto COLLSIION_CSPT_SET = "COLLISION_CSPT_SET";
constexpr auto GLOBAL_DCD_THICKNESS = "GLOBAL_DCD_THICKNESS";
constexpr auto ENABLE_SLIDING = "ENABLE_SLIDING";

constexpr auto ENABLE_DCD_REPULSION_FORCE = "ENABLE_DCD_REPULSION_FORCE";

constexpr auto PREVIOUS_COLLISION_TARGET = "PREVIOUS_COLLISION_TARGET";

constexpr auto TARGET_CELL_BUFFER = "TARGET_CELL_BUFFER";

constexpr auto PBD_USE_HARD_CONSTRAINT = "PBD_USE_HARD_CONSTRAINT";

constexpr auto SHAPE_MATCHING_REST_CM = "SHAPE_MATCHING_REST_CM";
constexpr auto SHAPE_MATCHING_WEIGHT_SUM = "SHAPE_MATCHING_WEIGHT_SUM";

constexpr auto SHAPE_MATCHING_SHAPE_OFFSET = "SHAPE_MATCHING_SHAPE_OFFSET";

constexpr auto SHAPE_MATCHING_MATRIX_BUFFER = "SHAPE_MATCHING_MATRIX_BUFFER";

constexpr auto VERTEX_BV = "VERTEX_BV";
constexpr auto VERTEX_BVH = "VERTEX_BV";
constexpr auto TRIANGLE_MESH_BV = "TRIANGLE_MESH_BV";
constexpr auto TRIANGLE_MESH_BVH = "TRIANGLE_MESH_BVH";

constexpr auto MESH_REORDER_KEYS = "MESH_REORDER_KEYS";
constexpr auto MESH_REORDER_INDICES = "MESH_REORDER_INDICES";
constexpr auto MESH_REORDER_VERTICES_BUFFER = "MESH_REORDER_VERTICES_BUFFER";
constexpr auto MESH_MAIN_AXIS = "MESH_MAIN_AXIS";
constexpr auto MESH_GLOBAL_BOUNDING_BOX = "MESH_GLOBAL_BOUNDING_BOX";

// constexpr auto DCD_COLLISIONS_MESH_COLLIDER = "DCD_COLLISION_MESH_COLLIDER";

enum category_c : int {
    shape_matching_constraint,
    long_range_attachment,
    vertex_pin_to_vertex_constraint,
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