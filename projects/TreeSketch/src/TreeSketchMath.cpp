#include <TreeSketchMath.h>
#include <cmath>
#include <cstdlib>

namespace zeno
{
    double random(const double min, const double max)
    {
        return min + static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX) * (max - min);
    }

    int random(const int min, const int max)
    {
        return min + std::rand() % (max - min + 1);
    }

    double degreeToRadian(const double degree)
    {
        return degree * 0.01745329251994329576923690768489;
    }

    zeno::vec4d lerp(const zeno::vec4d low, const zeno::vec4d high, const double t)
    {
        return (1.0 - t) * low + t * high;
    }

    zeno::mat4d operator*(const zeno::mat4d &lhs_m, const zeno::mat4d &rhs_m)
    {
        zeno::mat4d result{zeno::vec4d{}};
        for (std::size_t row{0}; row < 4; ++row)
        {
            for (std::size_t col{0}; col < 4; ++col)
            {
                double sum{0.0};
                for (std::size_t i{0}; i < 4; ++i)
                {
                    sum += lhs_m[row][i] * rhs_m[i][col];
                }
                result[row][col] = sum;
            }
        }
        return result;
    }

    zeno::vec4d operator*(const zeno::mat4d &m, const zeno::vec4d &v)
    {
        zeno::vec4d result{};
        for (std::size_t row{0}; row < 4; ++row)
        {
            double sum{0};
            for (std::size_t i{0}; i < 4; ++i)
            {
                sum += m[row][i] * v[i];
            }
            result[row] = sum;
        }
        return result;
    }

    zeno::mat4d rotate(const double radian, zeno::vec4d axis)
    {
        auto c{std::cos(radian)};
        auto s{std::sin(radian)};
        axis = zeno::normalize(axis);

        return {
            zeno::vec4d{
                (1.0 - c) * axis[0] * axis[0] + c,
                (1.0 - c) * axis[0] * axis[1] - axis[2] * s,
                (1.0 - c) * axis[0] * axis[2] + axis[1] * s,
                0.0,
            },
            zeno::vec4d{
                (1.0 - c) * axis[1] * axis[0] + axis[2] * s,
                (1.0 - c) * axis[1] * axis[1] + c,
                (1.0 - c) * axis[1] * axis[2] - axis[0] * s,
                0.0,
            },
            zeno::vec4d{
                (1.0 - c) * axis[2] * axis[0] - axis[1] * s,
                (1.0 - c) * axis[2] * axis[1] + axis[0] * s,
                (1.0 - c) * axis[2] * axis[2] + c,
                0.0,
            },
            zeno::vec4d{0.0, 0.0, 0.0, 1.0},
        };
    }

    zeno::mat4d transform_new_coord(
        const zeno::vec4d &new_origin,
        const zeno::vec4d &new_up)
    {
        auto new_y = zeno::normalize(zeno::vec3d{new_up[0], new_up[1], new_up[2]});
        zeno::vec3d old_z{0.0, 0.0, 1.0};
        auto new_x{zeno::normalize(zeno::cross(new_y, old_z))};
        auto new_z{zeno::cross(new_x, new_y)};
        return {
            zeno::vec4d{new_x[0], new_y[0], new_z[0], new_origin[0]},
            zeno::vec4d{new_x[1], new_y[1], new_z[1], new_origin[1]},
            zeno::vec4d{new_x[2], new_y[2], new_z[2], new_origin[2]},
            zeno::vec4d{0.0, 0.0, 0.0, 1.0},
        };
    }

    zeno::vec4d random_direction()
    {
        return zeno::normalize(zeno::vec4d{random(-1.0, 1.0), random(-1.0, 1.0), random(-1.0, 1.0), 0.0});
    }

    zeno::vec4d offset_direction(const zeno::vec4d &old_direction, const double offset_radian_min, const double offset_radian_max)
    {
        auto offset_radian{random(offset_radian_min, offset_radian_max)};
        auto rotation_axis{random_direction()};
        auto rotate_matrix{rotate(offset_radian, rotation_axis)};
        return zeno::normalize(rotate_matrix * old_direction);
    }

    std::vector<zeno::vec4d> calculate_turn_points(
        const zeno::vec4d &start, const zeno::vec4d &direction, const double length, const double radius,
        const int turn_points_num_min, const int turn_points_num_max,
        const double turn_points_offset_min, const double turn_points_offset_max)
    {
        std::vector<zeno::vec4d> turn_points{};
        auto turn_points_num{random(turn_points_num_min, turn_points_num_max)};

        for (auto i{0}; i < turn_points_num; ++i)
        {
            auto turn_point_offset{random(turn_points_offset_min, turn_points_offset_max)};

            auto turn_point{start + static_cast<double>(i + 1) / static_cast<double>(turn_points_num + 1) * length * direction};
            auto transform_matrix{transform_new_coord(turn_point, direction)};

            turn_point[0] = random(-1.0, 1.0);
            turn_point[1] = 0.0;
            turn_point[2] = random(-1.0, 1.0);
            turn_point[3] = 0.0;

            turn_point = zeno::normalize(turn_point);
            turn_point = turn_point_offset * turn_point;
            turn_point[3] = 1.0;

            turn_point = transform_matrix * turn_point;
            turn_points.push_back(turn_point);
        }

        return turn_points;
    }
};