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
roads::DynamicGrid<GridType> BuildGridFromPrimitive(zeno::PrimitiveObject* InPrimitive, const std::string& NxChannel, const std::string& NyChannel) {
    RoadsAssert(nullptr != InPrimitive, "[Roads] InPrimitive shouldn't be nullptr !");

    const auto Nx = InPrimitive->userData().get2<int>(NxChannel);
    const auto Ny = InPrimitive->userData().get2<int>(NyChannel);

    RoadsAssert(Nx * Ny <= InPrimitive->verts.size());

    roads::DynamicGrid<GridType> Grid(Nx, Ny);
    for (size_t i = 0; i < Nx * Ny; ++i) {
        Grid[i] = InPrimitive->verts[i];
    }
}

namespace {
    using namespace zeno;

    struct CalcPathCost_Simple : public zeno::reflect::IParameterAutoNode<CalcPathCost_Simple> {
        GENERATE_NODE_BODY(CalcPathCost_Simple);

        std::shared_ptr<zeno::PrimitiveObject> Primitive;
        DECLARE_INPUT_FIELD(Primitive, "Prim");
        DECLARE_OUTPUT_FIELD(Primitive, "Prim");

        std::string OutputChannel;
        DECLARE_INPUT_FIELD(OutputChannel, "OutputChannel", false, "", "path_cost");

        std::string SizeXChannel;
        DECLARE_INPUT_FIELD(SizeXChannel, "UserData_NxChannel", false, "", "nx");

        std::string SizeYChannel;
        DECLARE_INPUT_FIELD(SizeYChannel, "UserData_NyChannel", false, "", "ny");

        int Nx = 1;
        BINDING_PRIMITIVE_USERDATA(Primitive, Nx, SizeXChannel, true);

        zeno::AttrVector<vec3f> Test2 {};
        BINDING_PRIMITIVE_ATTRIBUTE(Primitive, Test2, "pos", zeno::reflect::EZenoPrimitiveAttr::VERT);

        void apply() override {
            zeno::log_info("aaaa: {}", AutoParameter->Nx);
            zeno::log_info("bbbb: {}", AutoParameter->Test2.size());
        }
    };
}
