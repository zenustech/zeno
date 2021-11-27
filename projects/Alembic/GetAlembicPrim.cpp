#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include "ABCTree.h"

namespace zeno {
namespace {

struct CountAlembicPrims : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        std::shared_ptr<PrimitiveObject> prim;
        int count = 0;
        abctree->visitPrims([&] (auto const &p) {
            count++;
        });
        set_output("count", std::make_shared<NumericObject>(count));
    }
};

ZENDEFNODE(CountAlembicPrims, {
    {{"ABCTree", "abctree"}},
    {{"int", "count"}},
    {},
    {"alembic"},
});


struct GetAlembicPrim : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        std::shared_ptr<PrimitiveObject> prim;
        int index = get_input<NumericObject>("index")->get<int>();
        abctree->visitPrims([&] (auto const &p) {
            if (index == 0) {
                prim = p;
                return false;
            }
            index--;
            return true;
        });
        if (!prim)
            throw Exception("index out of range in abctree");
        if (get_param<bool>("triangulate"))
            prim_triangulate(prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(GetAlembicPrim, {
    {{"ABCTree", "abctree"}, {"int", "index", "0"}},
    {{"PrimitiveObject", "prim"}},
    {{"bool", "triangulate", "1"}},
    {"alembic"},
});

} // namespace
} // namespace zeno
