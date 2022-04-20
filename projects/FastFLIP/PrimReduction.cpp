#include <zeno/zeno.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/parallel.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {

template <class T>
static T prim_reduce(PrimitiveObject *prim, std::string channel, std::string type)
{
    std::vector<T> const &temp = prim->attr<T>(channel);
    
    if(type==std::string("avg")){
        T total = parallel_reduce(tbb::blocked_range((size_t)1, temp.size()), temp[0],
        [&] (const auto &r, T total) -> T {
            for (size_t i = r.begin(); i < r.end(); i++) {
                total += temp[i];
            }
            return total;
        }, [&] (T i, T j) -> T { return i + j; });
        return total/(T)(temp.size());
    }
    if(type==std::string("max")){
        T total = parallel_reduce(tbb::blocked_range((size_t)1, temp.size()), temp[0],
        [&] (const auto &r, T total) -> T {
            for (size_t i = r.begin(); i < r.end(); i++) {
                total = zeno::max(total, temp[i]);
            }
            return total;
        }, [&] (T i, T j) -> T { return zeno::max(i, j); });
        return total;   
    }
    if(type==std::string("min")){
        T total = parallel_reduce(tbb::blocked_range((size_t)1, temp.size()), temp[0],
        [&] (const auto &r, T total) -> T {
            for (size_t i = r.begin(); i < r.end(); i++) {
                total = zeno::min(total, temp[i]);
            }
            return total;
        }, [&] (T i, T j) -> T { return zeno::min(i, j); });
        return total;
    }
    if(type==std::string("absmax"))
    {
        T total = parallel_reduce(tbb::blocked_range((size_t)1, temp.size()), zeno::abs(temp[0]),
        [&] (const auto &r, T total) -> T {
            for (size_t i = r.begin(); i < r.end(); i++) {
                total = zeno::max(total, zeno::abs(temp[i]));
            }
            return total;
        }, [&] (T i, T j) -> T { return zeno::max(zeno::abs(i), zeno::abs(j)); });
        return total;
    }
    return T(0);
}


struct PrimTbbReduction : zeno::INode {
    virtual void apply() override{
        auto prim = get_input<PrimitiveObject>("prim");
        auto attrToReduce = get_param<std::string>(("attr"));
        auto op = get_param<std::string>(("op"));
        zeno::NumericValue result;
        if (prim->attr_is<zeno::vec3f>(attrToReduce))
            result = prim_reduce<zeno::vec3f>(prim.get(), attrToReduce, op);
        else 
            result = prim_reduce<float>(prim.get(), attrToReduce, op);
        auto out = std::make_shared<zeno::NumericObject>();
        out->set(result);
        set_output("result", std::move(out));
    }
};
ZENDEFNODE(PrimTbbReduction,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "result",
    }, /* params: */ {
    {"string", "attr", "pos"},
    {"enum avg max min absmax", "op", "avg"},
    }, /* category: */ {
    "primitive",
    }});

}
