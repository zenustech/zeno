#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/NumericObject.h>
#include <zen/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
//#include <tbb/parallel_for.h>
//#include <tbb/parallel_reduce.h>

namespace zen {

template <class T>
static T prim_reduce(PrimitiveObject *prim, std::string channel, std::string type)
{
    std::vector<T> const &temp = prim->attr<T>(channel);
    
    if(type==std::string("avg")){
		T total = temp[0];
		for (int i = 1; i < temp.size(); i++) {
			total += temp[i];
		}
        return total/(T)(temp.size());
    }
    if(type==std::string("max")){
		T total = temp[0];
		for (int i = 1; i < temp.size(); i++) {
			total = zen::max(total, temp[i]);
		}
    }
    if(type==std::string("min")){
		T total = temp[0];
		for (int i = 1; i < temp.size(); i++) {
			total = zen::min(total, temp[i]);
		}
    }
    if(type==std::string("absmax"))
    {
        T total=zen::abs(temp[0]);
		for (int i = 1; i < temp.size(); i++) {
			total = zen::max(total, zen::abs(temp[i]));
		}
        return total;
    }
}


struct PrimitiveReduction : zen::INode {
    virtual void apply() override{
        auto prim = get_input("prim")->as<PrimitiveObject>();
        auto attrToReduce = std::get<std::string>(get_param("attr"));
        auto op = std::get<std::string>(get_param("op"));
        zen::NumericValue result;
        if (prim->attr_is<zen::vec3f>(attrToReduce))
            result = prim_reduce<zen::vec3f>(prim, attrToReduce, op);
        else 
            result = prim_reduce<float>(prim, attrToReduce, op);
        auto out = zen::IObject::make<zen::NumericObject>();
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
