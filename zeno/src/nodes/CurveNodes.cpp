#include <zeno/zeno.h>
#include <zeno/types/CurveObject.h>
#include <zeno/funcs/ParseObjectFromUi.h>
#include <sstream>
#include <array>

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

}
