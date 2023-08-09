
#include "zeno/zeno.h"

namespace {
    using namespace zeno;

    struct CalcPathCost_Simple : public INode {
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
    }
}
