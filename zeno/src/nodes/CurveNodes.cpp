#include <zeno/zeno.h>
#include <zeno/types/CurveObject.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/safe_at.h>

namespace zeno {

struct MakeCurve : zeno::INode {
    virtual void apply() override {
        auto curve = get_input<CurveObject>("curve:");
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
        prim->attr_visit(attrName, [&curve](auto &arr) {
            parallel_for_each(arr.begin(), arr.end(), [&] (auto &val) {
                val = curve->eval(val);
            });
        });
        set_output("prim", prim);
    }
};

ZENO_DEFNODE(EvalCurveOnPrimAttr)({
    {
        {"prim"},
        {"string", "attrName", "tmp"},
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



}
