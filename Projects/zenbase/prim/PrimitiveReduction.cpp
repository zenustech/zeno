#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/NumericObject.h>
#include <zen/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

namespace zenbase {

template <class T>
static T prim_reduce(PrimitiveObject *prim, std::string channel, std::string type)
{
    std::vector<T> temp = prim->attr<T>(channel);
    
    if(type==std::string("avg")){
        T start=temp[0];
        auto total = tbb::parallel_reduce(tbb::blocked_range<int>(1,temp.size()),
                start,
                [&](tbb::blocked_range<int> r, T running_total)
                {
                    for (int i=r.begin(); i<r.end(); ++i)
                    {
                        running_total += temp[i];
                    }

                    return running_total;
                }, [](auto a, auto b){return a+b; } );
        return total/(float)(temp.size());
    }
    if(type==std::string("max")){
        T start=temp[0];
        auto total = tbb::parallel_reduce(tbb::blocked_range<int>(1,temp.size()),
                start,
                [&](tbb::blocked_range<int> r, T running_total)
                {
                    for (int i=r.begin(); i<r.end(); ++i)
                    {
                        running_total = zen::max(running_total,temp[i]);
                    }

                    return running_total;
                }, [](auto a, auto b) { return zen::max(a,b); } );
        return total;
    }
    if(type==std::string("min")){
        T start=temp[0];
        auto total = tbb::parallel_reduce(tbb::blocked_range<int>(1,temp.size()),
                start,
                [&](tbb::blocked_range<int> r, T running_total)
                {
                    for (int i=r.begin(); i<r.end(); ++i)
                    {
                        running_total = zen::min(running_total,temp[i]);
                    }

                    return running_total;
                }, [](auto a, auto b) { return zen::min(a,b); } );
        return total;
    }
    if(type==std::string("absmax"))
    {
        T start=abs(temp[0]);
        auto total = tbb::parallel_reduce(tbb::blocked_range<int>(1,temp.size()),
                start,
                [&](tbb::blocked_range<int> r, T running_total)
                {
                    for (int i=r.begin(); i<r.end(); ++i)
                    {
                        running_total = zen::max(abs(running_total),abs(temp[i]));
                    }

                    return running_total;
                }, [](auto a, auto b) { return zen::max(abs(a),abs(b)); } );
        return total;
    }
}


struct PrimitiveReduction : zen::INode {
    virtual void apply() override{
        auto prim = get_input("prim")->as<PrimitiveObject>();
        auto attrToReduce = std::get<std::string>(get_param("attr"));
        auto op = std::get<std::string>(get_param("op"));
        zenbase::NumericValue result;
        if (prim->attr_is<zen::vec3f>(attrToReduce))
            result = prim_reduce<zen::vec3f>(prim, attrToReduce, op);
        else 
            result = prim_reduce<float>(prim, attrToReduce, op);
        auto out = zen::IObject::make<zenbase::NumericObject>();
        out->set(result);
        set_output("result", out);
    }
};
static int defPrimitiveReduction = zen::defNodeClass<PrimitiveReduction>("PrimitiveReduction",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "result",
    }, /* params: */ {
    {"string", "attr", "pos"},
    {"string", "op", "avg"},
    }, /* category: */ {
    "primitive",
    }});

}
