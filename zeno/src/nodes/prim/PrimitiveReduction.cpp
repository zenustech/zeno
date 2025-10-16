#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/IGeometryObject.h>
#include <zeno/utils/parallel_reduce.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
//#include <tbb/parallel_for.h>
//#include <tbb/parallel_reduce.h>

namespace zeno {

    template <class T>
    static T prim_reduce(std::vector<T> const& temp, std::string type)
    {
        if (type == std::string("avg")) {
            T total = zeno::parallel_reduce_array<T>(temp.size(), T(0), [&](size_t i) -> T { return temp[i]; },
                [&](T i, T j) -> T { return i + j; });
            return total / (T)(temp.size());
        }
        if (type == std::string("max")) {
            T total = zeno::parallel_reduce_array<T>(temp.size(), temp[0], [&](size_t i) -> T { return temp[i]; },
                [&](T i, T j) -> T { return zeno::max(i, j); });
            return total;
        }
        if (type == std::string("min")) {
            T total = zeno::parallel_reduce_array<T>(temp.size(), temp[0], [&](size_t i) -> T { return temp[i]; },
                [&](T i, T j) -> T { return zeno::min(i, j); });
            return total;
        }
        if (type == std::string("absmax")) {
            T total = zeno::parallel_reduce_array<T>(temp.size(), temp[0], [&](size_t i) -> T { return zeno::abs(temp[i]); },
                [&](T i, T j) -> T { return zeno::max(i, j); });
            return total;
        }
        return T(0);
    }


struct PrimReduction : zeno::INode {
    virtual void apply() override{
        auto prim = get_input_Geometry("prim");
        auto attrToReduce = get_input2_string("attrName");
        zeno::GeoAttrType type = prim->get_attr_type(ATTR_POINT, attrToReduce);
        auto op = get_input2_string("op");
        if (zeno::ATTR_VEC3 == type) {
            const std::vector<zeno::vec3f>& attrData = prim->get_vec3f_attr(ATTR_POINT, attrToReduce);
            zeno::vec3f result = prim_reduce<zeno::vec3f>(attrData, zsString2Std(op));
            ZImpl(set_primitive_output("result", result));
        }
        else {
            const std::vector<float>& attrData = prim->get_float_attr(ATTR_POINT, attrToReduce);
            float result = prim_reduce<float>(attrData, zsString2Std(op));
            ZImpl(set_primitive_output("result", result));
        }
    }
};
ZENDEFNODE(PrimReduction,{
    {
        {gParamType_Geometry, "prim", "", zeno::Socket_ReadOnly},
        {gParamType_String, "attrName", "pos"},
        {"enum avg max min absmax", "op", "avg"},
    },
    {
        {gParamType_AnyNumeric, "result"},
    },
    {},
    {"primitive"},
});
//struct PrimAttrScatter : zeno::INode {
//  virtual void apply() override{
//    auto prim = get_input<PrimitiveObject>("primFrom");
//    auto prim2 = get_input<PrimitiveObject>("primTo");
//    auto attrToReduce = get_input2<std::string>(("attrName"));
//    auto op = get_input2<std::string>(("op"));
//
//    set_output("result", std::move(prim));
//  }
//};
//ZENDEFNODE(PrimAttrScatter,{
//                              {
//                                    {"primFrom"},
//                                    {"primTo"},
//                                  {"string", "attrName", "pos"},
//                                  {"enum add max min absmax mul", "op", "add"},
//                              },
//                              {
//                                  {"prim"},
//                              },
//                              {},
//                              {"primitive"},
//                          });
struct PrimitiveBoundingBox : zeno::INode {
    virtual void apply() override{
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto &pos = prim->attr<zeno::vec3f>("pos");

        auto bmin = pos.size() ? pos[0] : vec3f(0);
        auto bmax = bmin;

        for (int i = 1; i < pos.size(); i++) {
            bmin = zeno::min(pos[i], bmin);
            bmax = zeno::max(pos[i], bmax);
        }

        // auto exwidth = ZImpl(get_param<float>("exWidth"));
        auto exwidth = ZImpl(has_input("exWidth")) ? ZImpl(get_input2<float>("exWidth")) : 0.f;
        if (exwidth) {
            bmin -= exwidth;
            bmax += exwidth;
        }
        ZImpl(set_output2("bmin", bmin));
        ZImpl(set_output2("bmax", bmax));
    }
};

ZENDEFNODE(PrimitiveBoundingBox,
    { /* inputs: */ {
    {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
    {gParamType_Float, "exWidth", "0"},
    }, /* outputs: */ {
    {gParamType_Vec3f, "bmin"}, {gParamType_Vec3f, "bmax"},
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});

}
