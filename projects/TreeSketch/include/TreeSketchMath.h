#ifndef __TREE_SKETCH_MATH_H__
#define __TREE_SKETCH_MATH_H__

#include <zeno/utils/vec.h>
#include <array>
#include <vector>

namespace zeno
{
    float random(const float min, const float max);

    float degreeToRadian(const float degree);

    zeno::vec4d lerp(const zeno::vec4d low, const zeno::vec4d high, const float t);

    using mat4d = std::array<zeno::vec4d, 4>; // row first

    zeno::mat4d operator*(const zeno::mat4d &lhs_m, const zeno::mat4d &rhs_m);

    zeno::vec4d operator*(const zeno::mat4d &m, const zeno::vec4d &v);

    zeno::mat4d rotate(const float radian, zeno::vec4d axis);

    zeno::mat4d transform_new_coord(const zeno::vec4d &new_origin, const zeno::vec4d &new_up);

    zeno::vec4d random_direction();

    zeno::vec4d offset_direction(
        const zeno::vec4d &old_direction, const float offset_radian_min, const float offset_radian_max);

    std::vector<zeno::vec4d> calculate_turn_points(
        const zeno::vec4d &start, const zeno::vec4d &direction, const float length, const float radius,
        const int turn_points_num_min, const int turn_points_num_max,
        const float turn_points_offset_min, const float turn_points_offset_max);
}

#endif