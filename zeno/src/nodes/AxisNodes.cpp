#include <zeno/zeno.h>
#include <zeno/types/AxisObject.h>

namespace zeno {
namespace {

struct ExtractAxis : zeno::INode {
    virtual void apply() override {
        auto p = ZImpl(get_input<AxisObject>("math"));
        ZImpl(set_output2("origin", p->origin));
        ZImpl(set_output2("axisX", p->axisX));
        ZImpl(set_output2("axisY", p->axisY));
        ZImpl(set_output2("axisZ", p->axisZ));
    }
};

ZENDEFNODE(ExtractAxis, {
    {
    {"AxisObject", "math"},
    },
    {
    {gParamType_Vec3f, "origin"},
    {gParamType_Vec3f, "axisX"},
    {gParamType_Vec3f, "axisY"},
    {gParamType_Vec3f, "axisZ"},
    },
    {},
    {"math"},
});

struct MakeAxis : zeno::INode {
    virtual void apply() override {
        auto origin = ZImpl(get_input2<zeno::vec3f>("origin"));
        auto axisX = ZImpl(get_input2<zeno::vec3f>("axisX"));
        auto axisY = ZImpl(get_input2<zeno::vec3f>("axisY"));
        auto axisZ = ZImpl(get_input2<zeno::vec3f>("axisZ"));
        auto p = std::make_unique<AxisObject>(origin, axisX, axisY, axisZ);
        auto by = ZImpl(get_param<std::string>("normalize"));
        if (by == "X")
            p->renormalizeByX();
        else if (by == "Y")
            p->renormalizeByY();
        else if (by == "Z")
            p->renormalizeByZ();
        ZImpl(set_output("math", std::move(p)));
    }
};

ZENDEFNODE(MakeAxis, {
    {
    {gParamType_Vec3f, "origin"},
    {gParamType_Vec3f, "axisX"},
    {gParamType_Vec3f, "axisY"},
    {gParamType_Vec3f, "axisZ"},
    },
    {
    {"AxisObject", "math"},
    },
    {
    {"enum off X Y Z", "normalize", "off"},
    },
    {"math"},
});

}
}
