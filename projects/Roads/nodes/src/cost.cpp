#include "zeno/zeno.h"
#include "zeno/utils/logger.h"
#include "zeno/PrimitiveObject.h"
#include "zeno/types/UserData.h"
#include "zeno/utils/PropertyVisitor.h"
#include "roads/roads.h"

template <typename ...Args>
inline void RoadsAssert(const bool Expr, const std::string& InMsg = "[Roads] Assert Failed", Args... args) {
    if (!Expr) {
        zeno::log_error(InMsg, args...);
    }
}

template <typename GridType = roads::Point>
roads::DynamicGrid<GridType> BuildGridFromPrimitive(const zeno::AttrVector<zeno::vec3f>& DataSource, int32_t Nx, int32_t Ny) {
    RoadsAssert(Nx * Ny <= DataSource.size());

    roads::DynamicGrid<GridType> Grid(Nx, Ny);
    for (size_t i = 0; i < Nx * Ny; ++i) {
        Grid[i] = roads::Point { DataSource[i][0], DataSource[i][0], DataSource[i][0] };
    }

    return Grid;
}

namespace {
    using namespace zeno;

    struct CalcPathCost_Simple : public zeno::reflect::IParameterAutoNode<CalcPathCost_Simple> {
        ZENO_GENERATE_NODE_BODY(CalcPathCost_Simple);

        std::shared_ptr<zeno::PrimitiveObject> Primitive;
        ZENO_DECLARE_INPUT_FIELD(Primitive, "Prim");
        ZENO_DECLARE_OUTPUT_FIELD(Primitive, "Prim");

        std::string OutputChannel;
        ZENO_DECLARE_INPUT_FIELD(OutputChannel, "OutputChannel", false, "", "path_cost");

        std::string SizeXChannel;
        ZENO_DECLARE_INPUT_FIELD(SizeXChannel, "UserData_NxChannel", false, "", "nx");

        std::string SizeYChannel;
        ZENO_DECLARE_INPUT_FIELD(SizeYChannel, "UserData_NyChannel", false, "", "ny");

        int Nx = 0;
        ZENO_BINDING_PRIMITIVE_USERDATA(Primitive, Nx, SizeXChannel, false);

        int Ny = 0;
        ZENO_BINDING_PRIMITIVE_USERDATA(Primitive, Ny, SizeYChannel, false);

        zeno::AttrVector<vec3f> PositionList {};
        ZENO_BINDING_PRIMITIVE_ATTRIBUTE(Primitive, PositionList, "pos", zeno::reflect::EZenoPrimitiveAttr::VERT);

        void apply() override {
//            BuildGridFromPrimitive(PositionList, Nx, Ny);
            zeno::log_info("aaa: {} {}", AutoParameter->Nx, AutoParameter->Ny);
        }
    };
}
