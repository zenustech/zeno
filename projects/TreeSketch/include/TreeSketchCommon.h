#ifndef __TREE_SKETCH_COMMON_H__
#define __TREE_SKETCH_COMMON_H__

#include <zeno/utils/vec.h>
#include <array>
#include <vector>

namespace zeno
{
    double random(const double min, const double max);

    int random(const int min, const int max);

    double degreeToRadian(const double degree);

    zeno::vec4d lerp(const zeno::vec4d low, const zeno::vec4d high, const double t);

    using mat4d = std::array<zeno::vec4d, 4>; // row first

    zeno::mat4d operator*(const zeno::mat4d &lhs_m, const zeno::mat4d &rhs_m);

    zeno::vec4d operator*(const zeno::mat4d &m, const zeno::vec4d &v);

    zeno::mat4d rotate(const double radian, zeno::vec4d axis);

    zeno::mat4d transform_new_coord(const zeno::vec4d &new_origin, const zeno::vec4d &new_up);

    zeno::vec4d random_direction();

    zeno::vec4d offset_direction(
        const zeno::vec4d &old_direction, const double offset_radian_min, const double offset_radian_max);

    std::vector<zeno::vec4d> calculate_turn_points(
        const zeno::vec4d &start, const zeno::vec4d &direction, const double length, const double radius,
        const int turn_points_num_min, const int turn_points_num_max,
        const double turn_points_offset_min, const double turn_points_offset_max);
}

#endif