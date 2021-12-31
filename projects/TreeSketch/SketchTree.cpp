#include <zeno/zeno.h>
#include <zeno/utils/vec.h>
#include <zeno/types/PrimitiveObject.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <vector>
#include <queue>
#include <iostream>

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

    double lerp(const double low, const double high, const double t)
    {
        return (1.0 - t) * low + t * high;
    }

    zeno::vec4d lerp(const zeno::vec4d low, const zeno::vec4d high, const double t)
    {
        return (1.0 - t) * low + t * high;
    }

    using mat4d = std::array<zeno::vec4d, 4>; // row first

    std::ostream &operator<<(std::ostream &os, const zeno::vec4d &v)
    {
        os << "[";
        for (std::size_t i{0}; i < 4; ++i)
        {
            os << v[i] << ' ';
        }
        os << "]\n";
        return os;
    }

    std::ostream &operator<<(std::ostream &os, const zeno::mat4d &m)
    {
        os << "[\n";
        for (std::size_t row{0}; row < 4; ++row)
        {
            os << "    [";
            for (std::size_t col{0}; col < 4; ++col)
            {
                os << m[row][col] << ' ';
            }
            os << "]\n";
        }
        os << "]\n";
        return os;
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

    struct TreeObj
        : zeno::IObject
    {
    private:
        struct BranchObj
        {
            zeno::vec4d _start;
            zeno::vec4d _direction;
            double _length;
            double _radius;
            std::vector<zeno::vec4d> _turn_points;
            bool _hasLeaf;
            std::vector<std::unique_ptr<BranchObj>> _children;

            BranchObj(const zeno::vec4d &start, const zeno::vec4d &direction,
                      const double length, const double radius,
                      const std::vector<zeno::vec4d> &turn_points)
                : _start{start}, _turn_points{turn_points}, _direction{direction}, _length{length},
                  _radius{radius}, _hasLeaf{false}, _children{} {}

            zeno::vec4d calculate_child_start(
                const double offset_start_min, const double offset_start_max)
            {
                auto offset_start{random(offset_start_min, offset_start_max)};

                auto mid{offset_start * (_turn_points.size() + 1) - 1.0};
                int low{static_cast<int>(std::floor(mid))};
                int high{low + 1};

                zeno::vec4d low_point;
                if (low == -1)
                {
                    low_point = _start;
                }
                else
                {
                    low_point = _turn_points[low];
                }

                zeno::vec4d high_point;
                if (high == _turn_points.size())
                {
                    high_point = _start + _length * _direction;
                }
                else
                {
                    high_point = _turn_points[high];
                }

                auto t{mid - static_cast<double>(low)};

                return lerp(low_point, high_point, t);
            }

        }; // BranchObj

        std::unique_ptr<BranchObj> _trunk;
        int _tree_level;

    public:
        TreeObj(
            const zeno::vec4d &start,
            const double offset_radian_min, const double offset_radian_max,
            const double length_min, const double length_max,
            const double radius_min, const double radius_max,
            const int turn_points_num_min, const int turn_points_num_max,
            const double turn_points_offset_min, const double turn_points_offset_max)
            : _tree_level{0}
        {
            std::srand(std::time(nullptr));

            zeno::vec4d up_direction{0.0, 1.0, 0.0, 0.0};
            auto direction{offset_direction(up_direction, offset_radian_min, offset_radian_max)};

            auto length{random(length_min, length_max)};

            auto radius{random(radius_min, radius_max)};

            auto turn_points{calculate_turn_points(
                start, direction, length, radius,
                turn_points_num_min, turn_points_num_max,
                turn_points_offset_min, turn_points_offset_max)};

            _trunk = std::make_unique<BranchObj>(
                start, direction, length, radius, turn_points);
        }

        void create_branchs(
            const int num_min, const int num_max,
            const double offset_start_min, const double offset_start_max,
            const double offset_radian_min, const double offset_radian_max,
            const double length_min, const double length_max,
            const double radius_min, const double radius_max,
            const int turn_points_num_min, const int turn_points_num_max,
            const double turn_points_offset_min, const double turn_points_offset_max)
        {
            std::queue<BranchObj *> branch_queue;
            std::queue<int> level_queue;
            branch_queue.push(_trunk.get());
            level_queue.push(0);

            while (!branch_queue.empty())
            {
                auto branchObj = branch_queue.front();
                branch_queue.pop();
                auto level = level_queue.front();
                level_queue.pop();

                if (level < _tree_level)
                {
                    for (const auto &child : branchObj->_children)
                    {
                        branch_queue.push(child.get());
                        level_queue.push(level + 1);
                    }
                }
                else if (level == _tree_level)
                {
                    auto num{random(num_min, num_max)};
                    for (auto i{0}; i < num; ++i)
                    {
                        auto start{branchObj->calculate_child_start(offset_start_min, offset_start_max)};

                        auto direction{offset_direction(branchObj->_direction, offset_radian_min, offset_radian_max)};

                        auto length{random(length_min, length_max)};

                        auto radius{random(radius_min, radius_max)};

                        auto turn_points{calculate_turn_points(
                            start, direction, length, radius,
                            turn_points_num_min, turn_points_num_max,
                            turn_points_offset_min, turn_points_offset_max)};

                        branchObj->_children.push_back(
                            std::make_unique<BranchObj>(
                                start, direction, length, radius, turn_points));
                    }
                }
            }
            ++_tree_level;
        }

        void set_leaves()
        {
            std::queue<BranchObj *> branch_queue;
            std::queue<int> level_queue;
            branch_queue.push(_trunk.get());
            level_queue.push(0);

            while (!branch_queue.empty())
            {
                auto branchObj = branch_queue.front();
                branch_queue.pop();
                auto level = level_queue.front();
                level_queue.pop();

                if (level < _tree_level)
                {
                    for (const auto &child : branchObj->_children)
                    {
                        branch_queue.push(child.get());
                        level_queue.push(level + 1);
                    }
                }
                else if (level == _tree_level)
                {
                    branchObj->_hasLeaf = true;
                }
            }
        }

        void toPrimitiveLines(zeno::PrimitiveObject *prim)
        {
            std::queue<BranchObj *> branch_queue;
            std::queue<int> level_queue;
            branch_queue.push(_trunk.get());
            level_queue.push(0);

            auto &pos = prim->add_attr<zeno::vec3f>("pos");

            while (!branch_queue.empty())
            {
                auto branchObj = branch_queue.front();
                branch_queue.pop();
                auto level = level_queue.front();
                level_queue.pop();

                int pre_pos_size = pos.size();

                const auto &start{branchObj->_start};
                pos.push_back(zeno::vec3f{static_cast<float>(start[0]), static_cast<float>(start[1]), static_cast<float>(start[2])});
                for (const auto &turn_point : branchObj->_turn_points)
                {
                    pos.push_back(zeno::vec3f{static_cast<float>(turn_point[0]), static_cast<float>(turn_point[1]), static_cast<float>(turn_point[2])});
                }
                const auto end{branchObj->_start + branchObj->_length * branchObj->_direction};
                pos.push_back(zeno::vec3f{static_cast<float>(end[0]), static_cast<float>(end[1]), static_cast<float>(end[2])});

                for (int pos_index = pre_pos_size; pos_index + 1 < pos.size(); ++pos_index)
                {
                    prim->lines.push_back(zeno::vec2i{pos_index, pos_index + 1});
                }

                if (level < _tree_level)
                {
                    for (const auto &child : branchObj->_children)
                    {
                        branch_queue.push(child.get());
                        level_queue.push(level + 1);
                    }
                }
            }
        }
    }; // struct TreeObj

    struct CreateTree
        : zeno::INode
    {
        virtual void apply() override
        {
            std::cout << "CreateTree::apply() called!\n";

            auto start_x = get_input2<double>("start_x");
            auto start_y = get_input2<double>("start_y");
            auto start_z = get_input2<double>("start_z");
            auto offset_degree_min = get_input2<double>("offset_degree_min");
            auto offset_degree_max = get_input2<double>("offset_degree_max");
            auto length_min = get_input2<double>("length_min");
            auto length_max = get_input2<double>("length_max");
            auto radius_min = get_input2<double>("radius_min");
            auto radius_max = get_input2<double>("radius_max");
            auto turn_points_num_min = get_input2<int>("turn_points_num_min");
            auto turn_points_num_max = get_input2<int>("turn_points_num_max");
            auto turn_points_offset_min = get_input2<double>("turn_points_offset_min");
            auto turn_points_offset_max = get_input2<double>("turn_points_offset_max");

            zeno::vec4d start{start_x, start_y, start_z, 1.0};
            auto offset_radian_min{degreeToRadian(offset_degree_min)};
            auto offset_radian_max{degreeToRadian(offset_degree_max)};

            auto treeObj = std::make_shared<TreeObj>(
                start,
                offset_radian_min, offset_radian_max,
                length_min, length_max,
                radius_min, radius_max,
                turn_points_num_min, turn_points_num_max,
                turn_points_offset_min, turn_points_offset_max);
            set_output("treeObj", std::move(treeObj));
        }
    }; // struct CreateTree

    ZENDEFNODE(
        CreateTree,
        {
            {
                {"double", "start_x", "0.0"},
                {"double", "start_y", "0.0"},
                {"double", "start_z", "0.0"},
                {"double", "offset_degree_min", "0.0"},
                {"double", "offset_degree_max", "0.0"},
                {"double", "length_min", "0.0"},
                {"double", "length_max", "0.0"},
                {"double", "radius_min", "0.0"},
                {"double", "radius_max", "0.0"},
                {"int", "turn_points_num_min", "0"},
                {"int", "turn_points_num_max", "0"},
                {"double", "turn_points_offset_min", "0.0"},
                {"double", "turn_points_offset_max", "0.0"},
            },
            {
                {"treeObj"},
            },
            {},
            {
                "TreeSketch",
            },
        } // CreateTree
    );

    struct TreeCreateBranchs
        : zeno::INode
    {
        virtual void apply() override
        {
            std::cout << "TreeCreateBranchs::apply() called!\n";

            auto treeObj = get_input<TreeObj>("treeObj");

            auto num_min = get_input2<int>("num_min");
            auto num_max = get_input2<int>("num_max");
            auto offset_start_min = get_input2<double>("offset_start_min");
            auto offset_start_max = get_input2<double>("offset_start_max");
            auto offset_degree_min = get_input2<double>("offset_degree_min");
            auto offset_degree_max = get_input2<double>("offset_degree_max");
            auto length_min = get_input2<double>("length_min");
            auto length_max = get_input2<double>("length_max");
            auto radius_min = get_input2<double>("radius_min");
            auto radius_max = get_input2<double>("radius_max");
            auto turn_points_num_min = get_input2<int>("turn_points_num_min");
            auto turn_points_num_max = get_input2<int>("turn_points_num_max");
            auto turn_points_offset_min = get_input2<double>("turn_points_offset_min");
            auto turn_points_offset_max = get_input2<double>("turn_points_offset_max");

            auto offset_radian_min{degreeToRadian(offset_degree_min)};
            auto offset_radian_max{degreeToRadian(offset_degree_max)};

            treeObj->create_branchs(
                num_min, num_max,
                offset_start_min, offset_start_max,
                offset_radian_min, offset_radian_max,
                length_min, length_max,
                radius_min, radius_max,
                turn_points_num_min, turn_points_num_max,
                turn_points_offset_min, turn_points_offset_max);
            set_output("treeObj", std::move(treeObj));
        }
    }; // struct TreeCreateBranchs

    ZENDEFNODE(
        TreeCreateBranchs,
        {
            {
                {"treeObj"},
                {"int", "num_min", "0"},
                {"int", "num_max", "0"},
                {"double", "offset_start_min", "0.0"},
                {"double", "offset_start_max", "0.0"},
                {"double", "offset_degree_min", "0.0"},
                {"double", "offset_degree_max", "0.0"},
                {"double", "length_min", "0.0"},
                {"double", "length_max", "0.0"},
                {"double", "radius_min", "0.0"},
                {"double", "radius_max", "0.0"},
                {"int", "turn_points_num_min", "0"},
                {"int", "turn_points_num_max", "0"},
                {"double", "turn_points_offset_min", "0.0"},
                {"double", "turn_points_offset_max", "0.0"},
            },
            {
                {"treeObj"},
            },
            {},
            {
                "TreeSketch",
            },
        } // TreeCreateBranchs
    );

    struct TreeSetLeaves
        : zeno::INode
    {
        virtual void apply() override
        {
            std::cout << "TreeSetLeaves::apply() called!\n";
            auto treeObj = get_input<TreeObj>("treeObj");
            treeObj->set_leaves();
            set_output("treeObj", std::move(treeObj));
        }
    }; // struct TreeSetLeaves

    ZENDEFNODE(
        TreeSetLeaves,
        {
            {
                {"treeObj"},
            },
            {
                {"treeObj"},
            },
            {},
            {
                "TreeSketch",
            },
        } // TreeSetLeaves
    );

    struct TreeToPrimitiveLines
        : zeno::INode
    {
        virtual void apply() override
        {
            std::cout << "TreeToPrimitiveLines::apply() called!\n";

            auto treeObj = get_input<TreeObj>("treeObj");

            auto prim = std::make_shared<zeno::PrimitiveObject>();
            treeObj->toPrimitiveLines(prim.get());

            set_output("prim", std::move(prim));
        }
    }; // struct TreeToPrimitiveLines

    ZENDEFNODE(
        TreeToPrimitiveLines,
        {
            {
                {"treeObj"},
            },
            {
                {"primitive", "prim"},
            },
            {},
            {
                "TreeSketch",
            },
        } // TreeToPrimitiveLines
    );
}; // namespace zeno