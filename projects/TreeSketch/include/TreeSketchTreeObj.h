#ifndef __TREE_SKETCH_TREE_OBJ_H__
#define __TREE_SKETCH_TREE_OBJ_H__

#include <TreeSketchMath.h>
#include <zeno/types/PrimitiveObject.h>
#include <memory>

namespace zeno
{
    struct TreeObj
        : zeno::IObject
    {
    private:
        struct BranchObj;

        std::unique_ptr<BranchObj> _trunk;
        int _tree_level;

    public:
        TreeObj(
            const zeno::vec4d &start,
            const float offset_radian_min, const float offset_radian_max,
            const float length_min, const float length_max,
            const float radius_min, const float radius_max,
            const int turn_points_num_min, const int turn_points_num_max,
            const float turn_points_offset_min, const float turn_points_offset_max);

        void create_branchs(
            const int num_min, const int num_max,
            const float offset_start_min, const float offset_start_max,
            const float offset_radian_min, const float offset_radian_max,
            const float length_min, const float length_max,
            const float radius_min, const float radius_max,
            const int turn_points_num_min, const int turn_points_num_max,
            const float turn_points_offset_min, const float turn_points_offset_max);

        void set_leaves();

        void to_primitive_lines(zeno::PrimitiveObject *prim);

    }; // struct TreeObj

    struct TreeObj::BranchObj
    {
        zeno::vec4d _start;
        zeno::vec4d _direction;
        float _length;
        float _radius;
        std::vector<zeno::vec4d> _turn_points;
        bool _hasLeaf;
        std::vector<std::unique_ptr<BranchObj>> _children;

        BranchObj(
            const zeno::vec4d &start, const zeno::vec4d &direction, const float length,
            const float radius, const std::vector<zeno::vec4d> &turn_points);

        zeno::vec4d calculate_child_start(const float offset_start_min, const float offset_start_max);

    }; // BranchObj
}

#endif