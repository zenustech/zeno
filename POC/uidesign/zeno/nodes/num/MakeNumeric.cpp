#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>

namespace {

struct NumericInt : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::NumericObject>();
        obj->set(get_param<int>("value"));
        set_output("value", std::move(obj));
    }
};

ZENDEFNODE(NumericInt, {
    {},
    {{"int", "value"}},
    {{"int", "value", "0"}},
    {"numeric"},
});


struct NumericIntVec2 : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::NumericObject>();
        auto x = get_param<int>("x");
        auto y = get_param<int>("y");
        obj->set(zeno::vec2i(x, y));
        set_output("vec2", std::move(obj));
    }
};

ZENDEFNODE(NumericIntVec2, {
    {},
    {{"vec2i", "vec2"}},
    {{"int", "x", "0"}, {"int", "y", "0"}},
    {"numeric"},
});


struct PackNumericIntVec2 : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::NumericObject>();
        auto x = get_input2<int>("x");
        auto y = get_input2<int>("y");
        obj->set(zeno::vec2i(x, y));
        set_output("vec2", std::move(obj));
    }
};

ZENDEFNODE(PackNumericIntVec2, {
    {{"int", "x", "0"}, {"int", "y", "0"}},
    {{"vec2i", "vec2"}},
    {},
    {"numeric"},
});


struct NumericIntVec3 : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::NumericObject>();
        auto x = get_param<int>("x");
        auto y = get_param<int>("y");
        auto z = get_param<int>("z");
        obj->set(zeno::vec3i(x, y, z));
        set_output("vec3", std::move(obj));
    }
};

ZENDEFNODE(NumericIntVec3, {
    {},
    {{"vec3i", "vec3"}},
    {{"int", "x", "0"}, {"int", "y", "0"}, {"int", "z", "0"}},
    {"numeric"},
});


struct NumericIntVec4 : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::NumericObject>();
        auto x = get_param<int>("x");
        auto y = get_param<int>("y");
        auto z = get_param<int>("z");
        auto w = get_param<int>("w");
        obj->set(zeno::vec4i(x, y, z, w));
        set_output("vec4", std::move(obj));
    }
};

ZENDEFNODE(NumericIntVec4, {
    {},
    {{"vec4f", "vec4"}},
    {{"float", "x", "0"}, {"float", "y", "0"},
     {"float", "z", "0"}, {"float", "w", "0"}},
    {"numeric"},
});


struct NumericFloat : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::NumericObject>();
        obj->set(get_param<float>("value"));
        set_output("value", std::move(obj));
    }
};

ZENDEFNODE(NumericFloat, {
    {},
    {{"float", "value"}},
    {{"float", "value", "0"}},
    {"numeric"},
});


struct NumericVec2 : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::NumericObject>();
        auto x = get_param<float>("x");
        auto y = get_param<float>("y");
        obj->set(zeno::vec2f(x, y));
        set_output("vec2", std::move(obj));
    }
};

ZENDEFNODE(NumericVec2, {
    {},
    {{"vec2f", "vec2"}},
    {{"float", "x", "0"}, {"float", "y", "0"}},
    {"numeric"},
});


struct NumericVec3 : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::NumericObject>();
        auto x = get_param<float>("x");
        auto y = get_param<float>("y");
        auto z = get_param<float>("z");
        
        obj->set(zeno::vec3f(x, y, z));
        set_output("vec3", std::move(obj));
    }
};

ZENDEFNODE(NumericVec3, {
    {},
    {{"vec3f", "vec3"}},
    {{"float", "x", "0"}, {"float", "y", "0"}, {"float", "z", "0"}},
    {"numeric"},
});


struct NumericVec4 : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zeno::NumericObject>();
        auto x = get_param<float>("x");
        auto y = get_param<float>("y");
        auto z = get_param<float>("z");
        auto w = get_param<float>("w");
        obj->set(zeno::vec4f(x, y, z, w));
        set_output("vec4", std::move(obj));
    }
};

ZENDEFNODE(NumericVec4, {
    {},
    {{"vec4f", "vec4"}},
    {{"float", "x", "0"}, {"float", "y", "0"},
     {"float", "z", "0"}, {"float", "w", "0"}},
    {"numeric"},
});

}
