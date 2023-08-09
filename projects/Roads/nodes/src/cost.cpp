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

    struct CalcPathCost_Simple : public zeno::reflect::IParameterAutoNode<CalcPathCost_Simple> {
        GENERATE_NODE_BODY(CalcPathCost_Simple);

        std::shared_ptr<zeno::PrimitiveObject> Primitive;
        DECLARE_INPUT_FIELD(Primitive, "prim");

        std::string OutputChannel;
        DECLARE_INPUT_FIELD(OutputChannel, "output_channel");

        DECLARE_OUTPUT_FIELD(Primitive, "prim");

        void apply() override;
    };

    void CalcPathCost_Simple::apply() {
    }
}
