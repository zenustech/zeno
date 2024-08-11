#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>

namespace {

struct NumericInt : zeno::INode {
    virtual void apply() override {
        set_primitive_output("value", get_param<int>("value"));
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
        auto x = get_param<int>("x");
        auto y = get_param<int>("y");
        set_primitive_output("vec2", zeno::vec2i(x, y));
    }
};

ZENDEFNODE(NumericIntVec2, {
    {},
    {{"vec2i", "vec2"}},
    {{"int", "x", "0"}, {"int", "y", "0"}},
    {"deprecated"},
});


struct PackNumericIntVec2 : zeno::INode {
    virtual void apply() override {
        auto x = get_input2<int>("x");
        auto y = get_input2<int>("y");
        set_primitive_output("vec2", zeno::vec2i(x, y));
    }
};

ZENDEFNODE(PackNumericIntVec2, {
    {{"int", "x", "0"}, {"int", "y", "0"}},
    {{"vec2i", "vec2"}},
    {},
    {"deprecated"},
});


struct NumericIntVec3 : zeno::INode {
    virtual void apply() override {
        auto x = get_param<int>("x");
        auto y = get_param<int>("y");
        auto z = get_param<int>("z");
        set_primitive_output("vec3", zeno::vec3i(x, y, z));
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
        auto x = get_param<int>("x");
        auto y = get_param<int>("y");
        auto z = get_param<int>("z");
        auto w = get_param<int>("w");
        set_primitive_output("vec4", zeno::vec4i(x, y, z, w));
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
        set_primitive_output("value", get_param<float>("value"));
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
        auto x = get_param<float>("x");
        auto y = get_param<float>("y");
        set_primitive_output("vec2", zeno::vec2f(x, y));
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
        auto x = get_param<float>("x");
        auto y = get_param<float>("y");
        auto z = get_param<float>("z");
        set_primitive_output("vec2", zeno::vec3f(x, y, z));
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
        auto x = get_param<float>("x");
        auto y = get_param<float>("y");
        auto z = get_param<float>("z");
        auto w = get_param<float>("w");
        set_primitive_output("vec2", zeno::vec4f(x, y, z, w));
    }
};

ZENDEFNODE(NumericVec4, {
    {},
    {{"vec4f", "vec4"}},
    {{"float", "x", "0"}, {"float", "y", "0"},
     {"float", "z", "0"}, {"float", "w", "0"}},
    {"numeric"},
});

struct PackNumericVecInt : zeno::INode {
    virtual void apply() override {
        auto _type = get_param<std::string>("type");
        auto x = get_input2<int>("x");
        auto y = get_input2<int>("y");
        auto z = get_input2<int>("z");
        auto w = get_input2<int>("w");
        if (_type == "int") {
            set_primitive_output("veci", x);
        } else if (_type == "vec2i") {
            set_primitive_output("veci", zeno::vec2i(x, y));
        } else if (_type == "vec3i") {
            set_primitive_output("veci", zeno::vec3i(x, y, z));
        } else if (_type == "vec4i") {
            set_primitive_output("veci", zeno::vec4f(x, y, z, w));
        }
    }
};

ZENDEFNODE(PackNumericVecInt, {
    {
        {"int", "x", "0"},
        {"int", "y", "0"},
        {"int", "z", "0"},
        {"int", "w", "0"},
    },
    {{"NumericObject","veci"}},
    {
        {"enum int vec2i vec3i vec4i", "type", "vec3i"},
    },
    {"numeric"},
});

struct PackNumericVec : zeno::INode {
    virtual void apply() override {
        auto _type = get_param<std::string>("type");
        auto x = get_input2<float>("x");
        auto y = get_input2<float>("y");
        auto z = get_input2<float>("z");
        auto w = get_input2<float>("w");
        if (_type == "float") {
            set_primitive_output("vec", x);
        } else if (_type == "vec2f") {
            set_primitive_output("vec", zeno::vec2f(x, y));
        } else if (_type == "vec3f") {
            set_primitive_output("vec", zeno::vec3f(x, y, z));
        } else if (_type == "vec4f") {
            set_primitive_output("vec", zeno::vec4f(x, y, z, w));
        }
    }
};

ZENDEFNODE(PackNumericVec, {
    {
        {"float", "x", "0"},
        {"float", "y", "0"},
        {"float", "z", "0"},
        {"float", "w", "0"},
    },
    {{"NumericObject","vec"}},
    {
        {"enum float vec2f vec3f vec4f", "type", "vec3f"},
    },
    {"numeric"},
});

}
