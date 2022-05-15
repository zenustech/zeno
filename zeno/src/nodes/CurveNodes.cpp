#include <zeno/zeno.h>
#include <zeno/types/CurveObject.h>
#include <sstream>
#include <array>

namespace zeno {

struct MakeCurve : zeno::INode {
    virtual void apply() override {
        auto curve = std::make_shared<zeno::CurveObject>();
        std::stringstream ss(get_param<std::string>("UI_MakeCurve"));
        size_t keycount = 0;
        ss >> keycount;
        for (size_t kid = 0; kid < keycount; kid++) {
            std::string key = "(uninit)";
            ss >> key;
            CurveData &cdat = curve->keys[key];
            int cyctype = 0;
            ss >> cyctype;
            cdat.cycleType = (CurveData::CycleType)cyctype;
            size_t count = 0;
            ss >> count;
            float input_min = 0, input_max = 1;
            float output_min = 0, output_max = 1;
            ss >> input_min >> input_max >> output_min >> output_max;
            for (size_t i = 0; i < count; i++) {
                float x = 0, y = 0;
                int cptype = 0;
                float x0 = 0, y0 = 0;
                float x1 = 0, y1 = 0;
                ss >> x >> y >> cptype >> x0 >> y0 >> x1 >> y1;
                cdat.addPoint(x, y, (CurveData::PointType)cptype, {x0, y0}, {x1, y1});
            }
        }
        set_output("curve", std::move(curve));
    }
};

ZENO_DEFNODE(MakeCurve)({
    {
    },
    {
        {"curve", "curve"},
    },
    {
        // FIXME: need have at least one param to prevent luzh bug!
        {"", "madebypengsensei", ""},
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
