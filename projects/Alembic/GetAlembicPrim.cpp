#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/prim_ops.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include "ABCTree.h"

namespace zeno {
namespace {

int count_alembic_prims(std::shared_ptr<zeno::ABCTree> abctree) {
    int count = 0;
    abctree->visitPrims([&] (auto const &p) {
        count++;
    });
    return count;
}

struct CountAlembicPrims : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        std::shared_ptr<PrimitiveObject> prim;
        int count = count_alembic_prims(abctree);
        set_output("count", std::make_shared<NumericObject>(count));
    }
};

ZENDEFNODE(CountAlembicPrims, {
    {{"ABCTree", "abctree"}},
    {{"int", "count"}},
    {},
    {"alembic"},
});

std::shared_ptr<PrimitiveObject> get_alembic_prim(std::shared_ptr<zeno::ABCTree> abctree, int index, bool triangulate) {
    std::shared_ptr<PrimitiveObject> prim;
    abctree->visitPrims([&] (auto const &p) {
        if (index == 0) {
            prim = p;
            return false;
        }
        index--;
        return true;
    });
    if (!prim) {
        throw Exception("index out of range in abctree");
    }
    if (triangulate) {
        prim_triangulate(prim.get());
    }
    return prim;
}
struct GetAlembicPrim : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        int index = get_input<NumericObject>("index")->get<int>();
        bool triangulate = get_param<bool>("triangulate");
        std::shared_ptr<PrimitiveObject> prim = get_alembic_prim(abctree, index, triangulate);
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(GetAlembicPrim, {
    {{"ABCTree", "abctree"}, {"int", "index", "0"}},
    {{"PrimitiveObject", "prim"}},
    {{"bool", "triangulate", "1"}},
    {"alembic"},
});

struct AllAlembicPrim : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        auto prims = std::make_shared<zeno::ListObject>();
        bool triangulate = get_param<bool>("triangulate");
        abctree->visitPrims([&] (auto const &p) {
            auto np = std::static_pointer_cast<PrimitiveObject>(p->clone());
            prims->arr.push_back(np);
        });
        auto outprim = primitive_merge(prims);
        if (triangulate) {
            prim_triangulate(outprim.get());
        }
        set_output("prim", std::move(outprim));
    }
};

ZENDEFNODE(AllAlembicPrim, {
    {{"ABCTree", "abctree"}},
    {{"PrimitiveObject", "prim"}},
    {{"bool", "triangulate", "1"}},
    {"alembic"},
});

} // namespace
} // namespace zeno
