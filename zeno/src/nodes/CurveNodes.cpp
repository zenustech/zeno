#include <zeno/zeno.h>
#include <zeno/types/CurveObject.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/safe_at.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/utils/arrayindex.h>

namespace zeno {

struct MakeCurve : zeno::INode {
    virtual void apply() override {
        auto curve = get_input<CurveObject>("curve");
        set_output("curve", curve);
    }
};

ZENO_DEFNODE(MakeCurve)({
    {
    },
    {
        {"curve", "curve"},
    },
    {
        //{"string", "madebypengsensei", "pybyes"},
        {"curve", "curve", ""},
    },
    {"curve"},
});

struct EvalCurve : zeno::INode {
    virtual void apply() override {
        auto curve = get_input<CurveObject>("curve");
        auto input = get_input2<NumericValue>("value");
        auto output = std::visit([&] (auto const &src) -> NumericValue {
            return curve->eval(src);
        }, input);
        set_output2("value", output);
    }
};

ZENO_DEFNODE(EvalCurve)({
    {
        {"float", "value"},
        {"curve", "curve"},
    },
    {
        {"float", "value"},
    },
    {},
    {"curve"},
});

struct EvalCurveOnPrimAttr : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto curve = get_input<CurveObject>("curve");
        auto attrName = get_input2<std::string>("attrName");
        auto dstName = get_input2<std::string>("dstName");
        prim->attr_visit(attrName, [&](auto &arr) {
            if (dstName.empty() || dstName == attrName) {
                parallel_for_each(arr.begin(), arr.end(), [&] (auto &val) {
                    val = curve->eval(val);
                });
            }
            else {
                using T = std::decay_t<decltype(arr[0])>;
                auto& dstAttr = prim->add_attr<T>(dstName);
                parallel_for(arr.size(), [&] (auto i) {
                    dstAttr[i] = curve->eval(arr[i]);
                });
            }
        });
        set_output("prim", prim);
    }
};

ZENO_DEFNODE(EvalCurveOnPrimAttr)({
    {
        {"prim"},
        {"string", "attrName", "tmp"},
        {"string", "dstName", ""},
        {"curve", "curve"},
    },
    {
        {"prim"},
    },
    {},
    {"curve"},
});


struct GetCurveControlPoint : zeno::INode {
    virtual void apply() override {
        auto curve = get_input<CurveObject>("curve");
        auto key = get_input2<std::string>("key");
        int i = get_input2<int>("index");

        auto &data = safe_at(curve->keys, key, "curve key");
        if (i < 0 || i >= data.cpbases.size())
            throw makeError<KeyError>(std::to_string(i), "out of range of " + std::to_string(data.cpbases.size()));
        set_output2("point_x", data.cpbases[i]);
        set_output2("point_y", data.cpoints[i].v);
        set_output2("left_handler", data.cpoints[i].left_handler);
        set_output2("right_handler", data.cpoints[i].right_handler);
    }
};

ZENO_DEFNODE(GetCurveControlPoint)({
    {
        {"curve", "curve"},
        {"string", "key", "x"},
        {"int", "index", "0"},
    },
    {
        {"float", "point_x"},
        {"float", "point_y"},
        {"vec2f", "left_handler"},
        {"vec2f", "right_handler"},
    },
    {},
    {"curve"},
});

struct UpdateCurveControlPoint : zeno::INode {
    virtual void apply() override {
        auto curve = get_input<CurveObject>("curve");
        auto key = get_input2<std::string>("key");
        int i = get_input2<int>("index");

        auto &data = safe_at(curve->keys, key, "curve key");
        if (i < 0 || i >= data.cpbases.size())
            throw makeError<KeyError>(std::to_string(i), "out of range of " + std::to_string(data.cpbases.size()));
        if (has_input("point_x"))
            data.cpbases[i] = get_input2<float>("point_x");
        if (has_input("point_y"))
            data.cpoints[i].v = get_input2<float>("point_y");
        if (has_input("left_handler"))
            data.cpoints[i].left_handler = get_input2<vec2f>("left_handler");
        if (has_input("right_handler"))
            data.cpoints[i].right_handler = get_input2<vec2f>("right_handler");

        set_output("curve", std::move(curve));
    }
};

ZENO_DEFNODE(UpdateCurveControlPoint)({
    {
        {"curve", "curve"},
        {"string", "key", "x"},
        {"int", "index", "0"},
        {"optional float", "point_x"},
        {"optional float", "point_y"},
        {"optional vec2f", "left_handler"},
        {"optional vec2f", "right_handler"},
    },
    {
        {"curve", "curve"},
    },
    {},
    {"curve"},
});

struct UpdateCurveCycleType : zeno::INode {
    virtual void apply() override {
        auto curve = get_input<CurveObject>("curve");
        auto key = get_input2<std::string>("key");
        auto type = get_input2<std::string>("type");
        auto typeIndex = array_index_safe({"CLAMP", "CYCLE", "MIRROR"}, type, "CycleType");
        if (key.empty()) {
            for (auto &[k, v]: curve->keys) {
                v.cycleType = static_cast<CurveData::CycleType>(typeIndex); 
            }
        } else {
            curve->keys.at(key).cycleType = static_cast<CurveData::CycleType>(typeIndex); 
        }
        set_output("curve", std::move(curve));
    }
};

ZENO_DEFNODE(UpdateCurveCycleType)({
    {
        {"curve", "curve"},
        {"string", "key", "x"},
        {"enum CLAMP CYCLE MIRROR", "type", "CLAMP"},
    },
    {
        {"curve", "curve"},
    },
    {},
    {"curve"},
});

struct UpdateCurveXYRange : zeno::INode {
    virtual void apply() override {
        auto curve = get_input<CurveObject>("curve");
        auto key = get_input2<std::string>("key");
        auto &data = curve->keys.at(key);
        auto rg = data.rg;
        if (has_input("x_from"))
            rg.xFrom = get_input2<float>("x_from");
        if (has_input("x_to"))
            rg.xTo = get_input2<float>("x_to");
        if (has_input("y_from"))
            rg.yFrom = get_input2<float>("y_from");
        if (has_input("y_to"))
            rg.yTo = get_input2<float>("y_to");
        data.updateRange(rg);

        set_output("curve", std::move(curve));
    }
};

ZENO_DEFNODE(UpdateCurveXYRange)({
    {
        {"curve", "curve"},
        {"string", "key", "x"},
        {"optional float", "x_from"},
        {"optional float", "x_to"},
        {"optional float", "y_from"},
        {"optional float", "y_to"},
    },
    {
        {"curve", "curve"},
    },
    {},
    {"curve"},
});

}
