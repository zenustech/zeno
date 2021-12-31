#ifndef __TREE_SKETCH_TREE_OBJ_H__
#define __TREE_SKETCH_TREE_OBJ_H__

#include <TreeSketchCommon.h>
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
            const double offset_radian_min, const double offset_radian_max,
            const double length_min, const double length_max,
            const double radius_min, const double radius_max,
            const int turn_points_num_min, const int turn_points_num_max,
            const double turn_points_offset_min, const double turn_points_offset_max);

        void create_branchs(
            const int num_min, const int num_max,
            const double offset_start_min, const double offset_start_max,
            const double offset_radian_min, const double offset_radian_max,
            const double length_min, const double length_max,
            const double radius_min, const double radius_max,
            const int turn_points_num_min, const int turn_points_num_max,
            const double turn_points_offset_min, const double turn_points_offset_max);

        void set_leaves();

        void to_primitive_lines(zeno::PrimitiveObject *prim);

    }; // struct TreeObj

    struct TreeObj::BranchObj
    {
        zeno::vec4d _start;
        zeno::vec4d _direction;
        double _length;
        double _radius;
        std::vector<zeno::vec4d> _turn_points;
        bool _hasLeaf;
        std::vector<std::unique_ptr<BranchObj>> _children;

        BranchObj(
            const zeno::vec4d &start, const zeno::vec4d &direction, const double length,
            const double radius, const std::vector<zeno::vec4d> &turn_points);

        zeno::vec4d calculate_child_start(const double offset_start_min, const double offset_start_max);

    }; // BranchObj
}

#endif