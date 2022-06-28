#include <zeno/zeno.h>
#include <zeno/types/CurveObject.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <zeno/types/PrimitiveObject.h>

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
            for (int i = 0; i < arr.size(); i++) {
                arr[i] = curve->eval(arr[i]);
            }
        });
        set_output("prim", prim);
    }
};

ZENO_DEFNODE(EvalCurveOnPrimAttr)({
    {
        {"prim"},
        {"string", "attrName", "pos"},
        {"curve", "curve"},
    },
    {
        {"prim"},
    },
    {},
    {"curve"},
});



}
