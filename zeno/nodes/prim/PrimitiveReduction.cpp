#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
//#include <tbb/parallel_for.h>
//#include <tbb/parallel_reduce.h>

namespace zeno {

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
			total = zeno::max(total, temp[i]);
		}
        return total;   
    }
    if(type==std::string("min")){
		T total = temp[0];
		for (int i = 1; i < temp.size(); i++) {
			total = zeno::min(total, temp[i]);
		}
        return total;
    }
    if(type==std::string("absmax"))
    {
        T total=zeno::abs(temp[0]);
		for (int i = 1; i < temp.size(); i++) {
			total = zeno::max(total, zeno::abs(temp[i]));
		}
        return total;
    }
    return T(0);
}


struct PrimitiveReduction : zeno::INode {
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
ZENDEFNODE(PrimitiveReduction,
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
