#include <openvdb/openvdb.h>
#include <openvdb/tools/GridTransformer.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>

namespace zeno {
struct TransformVDB : INode {
    template <typename T>
    double eulerToRadian(T angle) {
        return angle / 180.0 * M_PI;
    }

    template <typename GridT>
    void vdb_transform(GridT &grid, zeno::vec3f tran, zeno::vec3f euler, zeno::vec3f sca) {
        openvdb::math::Transform::Ptr transform = grid->transformPtr();
        transform->preScale(openvdb::math::Vec3d(sca[0], sca[1], sca[2]));
        transform->preRotate(eulerToRadian(euler[0]), openvdb::math::X_AXIS);
        transform->preRotate(eulerToRadian(euler[1]), openvdb::math::Y_AXIS);
        transform->preRotate(eulerToRadian(euler[2]), openvdb::math::Z_AXIS);
        transform->postTranslate(openvdb::math::Vec3d(tran[0], tran[1], tran[2]));
    }

    void apply() override {
        auto vdb = get_input<VDBGrid>("VDBGrid");
        auto tran = get_input2<zeno::vec3f>("translation");
        auto euler = get_input2<zeno::vec3f>("eulerXYZ");
        auto sca = get_input2<zeno::vec3f>("scaling");

        auto type = vdb->getType();
        if (type == "FloatGrid") {
            auto &grid = std::dynamic_pointer_cast<VDBFloatGrid>(vdb)->m_grid;
            vdb_transform(grid, tran, euler, sca);
        } else if (type == "Int32Grid") {
            auto &grid = std::dynamic_pointer_cast<VDBIntGrid>(vdb)->m_grid;
            vdb_transform(grid, tran, euler, sca);
        } else if (type == "Vec3fGrid") {
            auto &grid = std::dynamic_pointer_cast<VDBFloat3Grid>(vdb)->m_grid;
            vdb_transform(grid, tran, euler, sca);
        } else if (type == "Vec3IGrid") {
            auto &grid = std::dynamic_pointer_cast<VDBInt3Grid>(vdb)->m_grid;
            vdb_transform(grid, tran, euler, sca);
        } else if (type == "PointDataGrid") {
            auto &grid = std::dynamic_pointer_cast<VDBPointsGrid>(vdb)->m_grid;
            vdb_transform(grid, tran, euler, sca);
        } else {
            throw zeno::Exception("Bad VDB type.");
        }

        set_output("VDBGrid", vdb);
    }
};

ZENDEFNODE(TransformVDB, {/* inputs: */
                          {{gParamType_VDBGrid, "VDBGrid", "", zeno::Socket_ReadOnly},
                           {gParamType_Vec3f, "translation", "0, 0, 0"},
                           {gParamType_Vec3f, "eulerXYZ", "0, 0, 0"},
                           {gParamType_Vec3f, "scaling", "1, 1, 1"}},
                          /* outputs: */
                          {"VDBGrid"},
                          /* params: */
                          {},
                          /* category: */
                          {"openvdb"}});

} // namespace zeno