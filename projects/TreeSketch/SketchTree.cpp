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
    float random(const float min, const float max)
    {
        return min + std::rand() / static_cast<float>(RAND_MAX) * (max - min);
    }

    int random(const int min, const int max)
    {
        return min + std::rand() % (max - min + 1);
    }

    float degreeToRadian(const float degree)
    {
        return degree * 0.01745329251994329576923690768489;
    }

    using mat4f = std::array<zeno::vec4f, 4>; // row first

    zeno::vec4f operator*(const mat4f &m, const zeno::vec4f &v)
    {
        return zeno::vec4f{
            zeno::dot(m[0], v),
            zeno::dot(m[1], v),
            zeno::dot(m[2], v),
            zeno::dot(m[3], v),
        };
    }

    mat4f rotate(const float angle, const zeno::vec4f &v)
    {
        auto a{angle};
        auto c{std::cos(a)};
        auto s{std::sin(a)};

        auto axis{zeno::normalize(v)};
        auto temp{(1.0f - c) * axis};

        return mat4f{
            zeno::vec4f{
                temp[0] * axis[0] + c,
                temp[0] * axis[1] - axis[2] * s,
                temp[0] * axis[2] + axis[1] * s,
                0.0f,
            },
            zeno::vec4f{
                temp[1] * axis[0] + axis[2] * s,
                temp[1] * axis[1] + c,
                temp[1] * axis[2] - axis[0] * s,
                0.0f,
            },
            zeno::vec4f{
                temp[2] * axis[0] - axis[1] * s,
                temp[2] * axis[1] + axis[0] * s,
                temp[2] * axis[2] + c,
                0.0f,
            },
            zeno::vec4f{0.0f, 0.0f, 0.0f, 1.0f},
        };
    }

    zeno::vec4f random_direction()
    {
        zeno::vec4f direction{random(-1.0f, 1.0f), random(-1.0f, 1.0f), random(-1.0f, 1.0f), 0.0f};
        return zeno::normalize(direction);
    }

    zeno::vec4f offset_direction(const zeno::vec4f &old_direction, const float offset_radian)
    {
        auto rotation_axis{random_direction()};
        auto rotate_matrix{rotate(offset_radian, rotation_axis)};
        return zeno::normalize(rotate_matrix * old_direction);
    }

    struct TreeObj
        : zeno::IObject
    {
    private:
        struct BranchObj
        {
            zeno::vec4f _start;
            zeno::vec4f _direction;
            float _length;
            float _radius;
            bool _hasLeaf;
            std::vector<std::unique_ptr<BranchObj>> _children;

            BranchObj(const zeno::vec4f &start, const zeno::vec4f &direction,
                      const float length, const float radius)
                : _start{start}, _direction{direction}, _length{length},
                  _radius{radius}, _hasLeaf{false}, _children{} {}
        }; // BranchObj

        std::unique_ptr<BranchObj> _trunk;
        int _tree_level;

    public:
        TreeObj(
            const zeno::vec4f &start,
            const float offset_radian_min, const float offset_radian_max,
            const float length_min, const float length_max,
            const float radius_min, const float radius_max)
            : _tree_level{0}
        {
            std::srand(std::time(nullptr));
            auto offset_radian{random(offset_radian_min, offset_radian_max)};
            auto length{random(length_min, length_max)};
            auto radius{random(radius_min, radius_max)};

            zeno::vec4f up_direction{0.0, 1.0, 0.0, 0.0};
            zeno::vec4f direction{offset_direction(up_direction, offset_radian)};

            _trunk = std::make_unique<BranchObj>(start, direction,
                                                 length, radius);
        }

        void create_branchs(
            const int num_min, const int num_max,
            const float offset_start_min, const float offset_start_max,
            const float offset_radian_min, const float offset_radian_max,
            const float length_min, const float length_max,
            const float radius_min, const float radius_max)
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
                        auto offset_start{random(offset_start_min, offset_start_max)};
                        auto offset_radian{random(offset_radian_min, offset_radian_max)};
                        auto length{random(length_min, length_max)};
                        auto radius{random(radius_min, radius_max)};

                        auto start{branchObj->_start + offset_start * branchObj->_length * branchObj->_direction};
                        auto direction{offset_direction(branchObj->_direction, offset_radian)};

                        branchObj->_children.push_back(
                            std::make_unique<BranchObj>(start, direction,
                                                        length, radius));
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

        void display()
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

                std::cout << "level: " << level << '\n'
                          << "start: "
                          << branchObj->_start[0] << ' '
                          << branchObj->_start[1] << ' '
                          << branchObj->_start[2] << ' '
                          << branchObj->_start[3] << '\n'
                          << "direction: "
                          << branchObj->_direction[0] << ' '
                          << branchObj->_direction[1] << ' '
                          << branchObj->_direction[2] << ' '
                          << branchObj->_direction[3] << '\n'
                          << "length: " << branchObj->_length << '\n'
                          << "radius: " << branchObj->_radius << '\n'
                          << "hasLeaf: " << branchObj->_hasLeaf << '\n';

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

                const auto &start{branchObj->_start};
                auto end{branchObj->_start + branchObj->_length * branchObj->_direction};
                pos.push_back(zeno::vec3f{start[0], start[1], start[2]});
                pos.push_back(zeno::vec3f{end[0], end[1], end[2]});
                auto pos_size = pos.size();
                prim->lines.push_back(zeno::vec2i(pos_size - 2, pos_size - 1));

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

            auto start_x = get_param<float>("start_x");
            auto start_y = get_param<float>("start_y");
            auto start_z = get_param<float>("start_z");
            auto offset_degree_min = get_param<float>("offset_degree_min");
            auto offset_degree_max = get_param<float>("offset_degree_max");
            auto length_min = get_param<float>("length_min");
            auto length_max = get_param<float>("length_max");
            auto radius_min = get_param<float>("radius_min");
            auto radius_max = get_param<float>("radius_max");

            zeno::vec4f start{start_x, start_y, start_z, 1.0};
            auto offset_radian_min{degreeToRadian(offset_degree_min)};
            auto offset_radian_max{degreeToRadian(offset_degree_max)};

            auto treeObj = std::make_shared<TreeObj>(
                start,
                offset_radian_min, offset_radian_max,
                length_min, length_max,
                radius_min, radius_max);
            set_output("treeObj", std::move(treeObj));
        }
    }; // struct CreateTree

    ZENDEFNODE(
        CreateTree,
        {
            {},
            {
                {"treeObj"},
            },
            {
                {"float", "start_x", "0.0"},
                {"float", "start_y", "0.0"},
                {"float", "start_z", "0.0"},
                {"float", "offset_degree_min", "0.0"},
                {"float", "offset_degree_max", "0.0"},
                {"float", "length_min", "0.0"},
                {"float", "length_max", "0.0"},
                {"float", "radius_min", "0.0"},
                {"float", "radius_max", "0.0"},
            },
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

            auto num_min = get_param<int>("num_min");
            auto num_max = get_param<int>("num_max");
            auto offset_start_min = get_param<float>("offset_start_min");
            auto offset_start_max = get_param<float>("offset_start_max");
            auto offset_degree_min = get_param<float>("offset_degree_min");
            auto offset_degree_max = get_param<float>("offset_degree_max");
            auto length_min = get_param<float>("length_min");
            auto length_max = get_param<float>("length_max");
            auto radius_min = get_param<float>("radius_min");
            auto radius_max = get_param<float>("radius_max");

            auto offset_radian_min{degreeToRadian(offset_degree_min)};
            auto offset_radian_max{degreeToRadian(offset_degree_max)};

            treeObj->create_branchs(
                num_min, num_max,
                offset_start_min, offset_start_max,
                offset_radian_min, offset_radian_max,
                length_min, length_max,
                radius_min, radius_max);
            set_output("treeObj", std::move(treeObj));
        }
    }; // struct TreeCreateBranchs

    ZENDEFNODE(
        TreeCreateBranchs,
        {
            {
                {"treeObj"},
            },
            {
                {"treeObj"},
            },
            {
                {"int", "num_min", "0"},
                {"int", "num_max", "0"},
                {"float", "offset_start_min", "0.0"},
                {"float", "offset_start_max", "0.0"},
                {"float", "offset_degree_min", "0.0"},
                {"float", "offset_degree_max", "0.0"},
                {"float", "length_min", "0.0"},
                {"float", "length_max", "0.0"},
                {"float", "radius_min", "0.0"},
                {"float", "radius_max", "0.0"},
            },
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

    struct TreeDisplay
        : zeno::INode
    {
        virtual void apply() override
        {
            std::cout << "TreeDisplay::apply() called!\n";
            auto treeObj = get_input<TreeObj>("treeObj");
            treeObj->display();
        }
    }; // struct TreeDisplay

    ZENDEFNODE(
        TreeDisplay,
        {
            {
                {"treeObj"},
            },
            {},
            {},
            {
                "TreeSketch",
            },
        } // TreeDisplay
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