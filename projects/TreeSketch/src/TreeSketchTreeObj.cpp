#include <TreeSketchTreeObj.h>
#include <cmath>
#include <ctime>
#include <queue>
#include <zeno/types/ListObject.h>
#include <iostream>


namespace zeno
{
    TreeObj::BranchObj::BranchObj(
        const zeno::vec4d &start, const zeno::vec4d &direction, const float length, const float radius,
        const std::vector<zeno::vec4d> &turn_points)
        : _start{start}, _direction{direction}, _length{length}, _radius{radius},
          _turn_points{turn_points}, _hasLeaf{false}, _children{} {}

    zeno::vec4d TreeObj::BranchObj::calculate_child_start(
        const float offset_start_min, const float offset_start_max)
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

        auto t{mid - static_cast<float>(low)};

        return lerp(low_point, high_point, t);
    }

    TreeObj::TreeObj(
        const zeno::vec4d &start,
        const float offset_radian_min, const float offset_radian_max,
        const float length_min, const float length_max,
        const float radius_min, const float radius_max,
        const int turn_points_num_min, const int turn_points_num_max,
        const float turn_points_offset_min, const float turn_points_offset_max)
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

    void TreeObj::create_branchs(
        const int num_min, const int num_max,
        const float offset_start_min, const float offset_start_max,
        const float offset_radian_min, const float offset_radian_max,
        const float length_min, const float length_max,
        const float radius_min, const float radius_max,
        const int turn_points_num_min, const int turn_points_num_max,
        const float turn_points_offset_min, const float turn_points_offset_max)
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

    void TreeObj::set_leaves()
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

    void TreeObj::to_primitive_lines(zeno::PrimitiveObject *prim, std::vector<std::shared_ptr<zeno::PrimitiveObject>> & primList)
    {
        std::queue<BranchObj *> branch_queue;
        std::queue<int> level_queue;
        branch_queue.push(_trunk.get());
        level_queue.push(0);

        //auto &pos = prim->add_attr<zeno::vec3f>("pos");

        primList.resize(0);

        while (!branch_queue.empty())
        {
            
            auto branchObj = branch_queue.front();
            branch_queue.pop();
            auto level = level_queue.front();
            level_queue.pop();
            //current branch 
            primList.push_back(std::make_shared<zeno::PrimitiveObject>());
            auto &curr_prim = primList.back();
            auto &pos = curr_prim->add_attr<zeno::vec3f>("pos");
            auto &radius = curr_prim->add_attr<float>("radius");
            int pre_pos_size = pos.size();

            const auto &start{branchObj->_start};
            pos.push_back(zeno::vec3f{static_cast<float>(start[0]), static_cast<float>(start[1]), static_cast<float>(start[2])});
            float r = branchObj->_radius;
            for (const auto &turn_point : branchObj->_turn_points)
            {
                pos.push_back(zeno::vec3f{static_cast<float>(turn_point[0]), static_cast<float>(turn_point[1]), static_cast<float>(turn_point[2])});
                radius.push_back(r);
                r*=0.9;
            }
            const auto end{branchObj->_start + branchObj->_length * branchObj->_direction};
            pos.push_back(zeno::vec3f{static_cast<float>(end[0]), static_cast<float>(end[1]), static_cast<float>(end[2])});
            radius.push_back(r);
            for (int pos_index = 0; pos_index + 1 < pos.size(); ++pos_index)
            {
                //prim->lines.push_back(zeno::vec2i{pos_index, pos_index + 1});
                curr_prim->lines->push_back(zeno::vec2i{pos_index, pos_index + 1});
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

}