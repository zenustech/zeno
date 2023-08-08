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

    struct CalcPathCostParameter : public zeno::reflect::NodeParameterBase {
        explicit CalcPathCostParameter(INode* Node) : zeno::reflect::NodeParameterBase(Node) {
            RunInputHooks();
        }

        std::shared_ptr<zeno::PrimitiveObject> Primitive;
        zeno::reflect::InputField<CalcPathCostParameter, decltype(Primitive)> PrimitiveField {*this, Primitive, "prim", false};

        std::string OutputChannel;
        zeno::reflect::InputField<CalcPathCostParameter, decltype(OutputChannel)> OutputChannelField { *this, OutputChannel, "output_channel" };
    };

    struct CalcPathCost_Simple : public zeno::reflect::IAutoNode<CalcPathCostParameter> {
        void apply() override;
    };

    ZENDEFNODE(
        CalcPathCost_Simple,
        {
            {
                { "prim" },
                { "string", "output_channel", "path_cost" },
            },
            {},
            {},
            {
                "Unreal"
            }
        }
    )

    void CalcPathCost_Simple::apply() {
        zeno::log_info("aaaaaaaaaaa: {}", AutoParameter->OutputChannel);
    }
}
