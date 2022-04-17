#include <zeno/zeno.h>
#include <zeno/types/AxisObject.h>

namespace zeno {
namespace {

struct ExtractAxis : zeno::INode {
    virtual void apply() override {
        auto p = get_input<AxisObject>("axis");
        set_output2("origin", p->origin);
        set_output2("axisX", p->axisX);
        set_output2("axisY", p->axisY);
        set_output2("axisZ", p->axisZ);
    }
};

ZENDEFNODE(ExtractAxis, {
    {
    {"AxisObject", "axis"},
    },
    {
    {"vec3f", "origin"},
    {"vec3f", "axisX"},
    {"vec3f", "axisY"},
    {"vec3f", "axisZ"},
    },
    {},
    {"axis"},
});

struct MakeAxis : zeno::INode {
    virtual void apply() override {
        auto origin = get_input2<vec3f>("origin");
        auto axisX = get_input2<vec3f>("axisX");
        auto axisY = get_input2<vec3f>("axisY");
        auto axisZ = get_input2<vec3f>("axisZ");
        auto p = std::make_shared<AxisObject>(origin, axisX, axisY, axisZ);
        auto by = get_param<std::string>("normalize");
        if (by == "X")
            p->renormalizeByX();
        else if (by == "Y")
            p->renormalizeByY();
        else if (by == "Z")
            p->renormalizeByZ();
        set_output("axis", std::move(p));
    }
};

ZENDEFNODE(MakeAxis, {
    {
    {"vec3f", "origin"},
    {"vec3f", "axisX"},
    {"vec3f", "axisY"},
    {"vec3f", "axisZ"},
    },
    {
    {"AxisObject", "axis"},
    },
    {
    {"enum off X Y Z", "normalize", "off"},
    },
    {"axis"},
});

}
}
