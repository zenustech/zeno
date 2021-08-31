#include <zeno/zeno.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {

template <class ValT, class GetF, class SumF>
inline ValT omp_parallel_reduce(size_t num, ValT init, GetF const &get, SumF const &sum) {
    constexpr int nproc = 512;
    ValT tls[nproc];
    for (int p = 0; p < nproc; p++) {
        tls[p] = init;
    }
#pragma omp parallel for
    for (int p = 0; p < nproc; p++) {
        size_t i0 = num / nproc * p;
        size_t i1 = p == nproc - 1 ? num : num / nproc * (p + 1);
        for (size_t i = i0; i < i1; i++) {
            tls[p] = sum(tls[p], get(i));
        }
    }
    ValT ret = init;
    for (int p = 0; p < nproc; p++) {
        ret = sum(ret, tls[p]);
    }
    return ret;
}

template <class T>
static T prim_reduce_omp(PrimitiveObject *prim, std::string channel, std::string type)
{
    std::vector<T> const &temp = prim->attr<T>(channel);
    
    if(type==std::string("avg")){
        T total = omp_parallel_reduce<T>(temp.size(), T(0), [&] (size_t i) -> T { return temp[i]; },
        [&] (T i, T j) -> T { return i + j; });
        return total/(T)(temp.size());
    }
    if(type==std::string("max")){
        T total = omp_parallel_reduce<T>(temp.size(), temp[0], [&] (size_t i) -> T { return temp[i]; },
        [&] (T i, T j) -> T { return zeno::max(i, j); });
        return total;   
    }
    if(type==std::string("min")){
        T total = omp_parallel_reduce<T>(temp.size(), temp[0], [&] (size_t i) -> T { return temp[i]; },
        [&] (T i, T j) -> T { return zeno::min(i, j); });
        return total;
    }
    if(type==std::string("absmax")){
        T total = omp_parallel_reduce<T>(temp.size(), temp[0], [&] (size_t i) -> T { return temp[i]; },
        [&] (T i, T j) -> T { return zeno::min(zeno::abs(i), zeno::abs(j)); });
        return total;
    }
    return T(0);
}

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
        auto attrToReduce = std::get<std::string>(get_param("attr"));
        auto op = std::get<std::string>(get_param("op"));
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


struct PrimOmpReduction : zeno::INode {
    virtual void apply() override{
        auto prim = get_input<PrimitiveObject>("prim");
        auto attrToReduce = std::get<std::string>(get_param("attr"));
        auto op = std::get<std::string>(get_param("op"));
        zeno::NumericValue result;
        if (prim->attr_is<zeno::vec3f>(attrToReduce))
            result = prim_reduce_omp<zeno::vec3f>(prim.get(), attrToReduce, op);
        else 
            result = prim_reduce_omp<float>(prim.get(), attrToReduce, op);
        auto out = std::make_shared<zeno::NumericObject>();
        out->set(result);
        set_output("result", std::move(out));
    }
};
ZENDEFNODE(PrimOmpReduction,
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
