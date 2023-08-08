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
roads::DynamicGrid<GridType> BuildGridFromPrimitive(zeno::PrimitiveObject* InPrimitive) {
    RoadsAssert(nullptr != InPrimitive, "[Roads] InPrimitive shouldn't be nullptr !");

    const auto Nx = InPrimitive->userData().get2<int>("nx");
    const auto Ny = InPrimitive->userData().get2<int>("ny");

    RoadsAssert(Nx * Ny <= InPrimitive->verts.size());

    roads::DynamicGrid<GridType> Grid(Nx, Ny);
    for (size_t i = 0; i < Nx * Ny; ++i) {
        Grid[i] = InPrimitive->verts[i];
    }
}

namespace {
    using namespace zeno;

    struct CalcPathCostParameter : public zeno::reflect::INodeParameterObject<CalcPathCostParameter> {
        GENERATE_PARAMETER_BODY(CalcPathCostParameter);

        std::shared_ptr<zeno::PrimitiveObject> Primitive;
        DECLARE_INPUT_FIELD(Primitive, "prim");

        std::string OutputChannel;
        DECLARE_INPUT_FIELD(OutputChannel, "output_channel");

        std::string OutputTest;
        DECLARE_OUTPUT_FIELD(OutputTest, "test");
    };

    struct CalcPathCost_Simple : public zeno::reflect::IAutoNode<CalcPathCost_Simple, CalcPathCostParameter> {
        GENERATE_AUTONODE_BODY(CalcPathCost_Simple);

        void apply() override;
    };

    void CalcPathCost_Simple::apply() {
        zeno::log_info("aaaaaaaaaaa: {}", AutoParameter->OutputChannel);
    }
}
