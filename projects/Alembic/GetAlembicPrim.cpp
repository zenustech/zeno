#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
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

std::shared_ptr<PrimitiveObject> primitive_merge(std::shared_ptr<zeno::ListObject> list) {
    auto outprim = std::make_shared<PrimitiveObject>();

    size_t len = 0;
    size_t poly_len = 0;

    //fix pyb
    for (auto const &prim: list->get<std::shared_ptr<PrimitiveObject>>()) {
        prim->foreach_attr([&] (auto const &key, auto const &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            outprim->add_attr<T>(key);
        });
    }
    //fix pyb

    for (auto const &prim: list->get<std::shared_ptr<PrimitiveObject>>()) {
        const auto base = outprim->size();
        prim->foreach_attr([&] (auto const &key, auto const &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            //fix pyb
            auto &outarr = outprim->attr<T>(key);
            outarr.insert(outarr.end(), std::begin(arr), std::end(arr));
            //for (auto const &val: arr) outarr.push_back(val);
            //end fix pyb
        });
        for (auto const &idx: prim->points) {
            outprim->points.push_back(idx + len);
        }
        for (auto const &idx: prim->lines) {
            outprim->lines.push_back(idx + len);
        }
        for (auto const &idx: prim->tris) {
            outprim->tris.push_back(idx + len);
        }
        for (auto const &idx: prim->quads) {
            outprim->quads.push_back(idx + len);
        }
        for (auto const &idx: prim->loops) {
            outprim->loops.push_back(idx + len);
        }
        size_t sub_poly_len = 0;
        for (auto const &poly: prim->polys) {
            sub_poly_len = std::max(sub_poly_len, (size_t)(poly.first + poly.second));
            outprim->polys.emplace_back(poly.first + poly_len, poly.second);
        }
        poly_len += sub_poly_len;
        len += prim->size();
        //fix pyb
        outprim->resize(len);
    }

    return outprim;
}

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
