#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>

namespace zeno {

struct NumericInt : INode {
    virtual void apply() override {
        ZImpl(set_primitive_output("value", ZImpl(get_param<int>("value"))));
    }
};

ZENDEFNODE(NumericInt, {
    {},
    {{gParamType_Int, "value"}},
    {{gParamType_Int, "value", "0"}},
    {"numeric"},
});


struct NumericIntVec2 : INode {
    virtual void apply() override {
        auto x = ZImpl(get_param<int>("x"));
        auto y = ZImpl(get_param<int>("y"));
        ZImpl(set_primitive_output("vec2", vec2i(x, y)));
    }
};

ZENDEFNODE(NumericIntVec2, {
    {},
    {{gParamType_Vec2i, "vec2"}},
    {{gParamType_Int, "x", "0"}, {gParamType_Int, "y", "0"}},
    {"deprecated"},
});


struct PackNumericIntVec2 : INode {
    virtual void apply() override {
        auto x = ZImpl(get_input2<int>("x"));
        auto y = ZImpl(get_input2<int>("y"));
        ZImpl(set_primitive_output("vec2", vec2i(x, y)));
    }
};

ZENDEFNODE(PackNumericIntVec2, {
    {{gParamType_Int, "x", "0"}, {gParamType_Int, "y", "0"}},
    {{"vec2i", "vec2"}},
    {},
    {"deprecated"},
});


struct NumericIntVec3 : INode {
    virtual void apply() override {
        auto x = ZImpl(get_param<int>("x"));
        auto y = ZImpl(get_param<int>("y"));
        auto z = ZImpl(get_param<int>("z"));
        ZImpl(set_primitive_output("vec3", vec3i(x, y, z)));
    }
};

ZENDEFNODE(NumericIntVec3, {
    {},
    {{gParamType_Vec3i, "vec3"}},
    {{gParamType_Int, "x", "0"}, {gParamType_Int, "y", "0"}, {gParamType_Int, "z", "0"}},
    {"numeric"},
});


struct NumericIntVec4 : INode {
    virtual void apply() override {
        auto x = ZImpl(get_param<int>("x"));
        auto y = ZImpl(get_param<int>("y"));
        auto z = ZImpl(get_param<int>("z"));
        auto w = ZImpl(get_param<int>("w"));
        ZImpl(set_primitive_output("vec4", vec4i(x, y, z, w)));
    }
};

ZENDEFNODE(NumericIntVec4, {
    {},
    {{gParamType_Vec4i, "vec4"}},
    {{gParamType_Float, "x", "0"}, {gParamType_Float, "y", "0"},
     {gParamType_Float, "z", "0"}, {gParamType_Float, "w", "0"}},
    {"numeric"},
});


struct NumericFloat : INode {
    virtual void apply() override {
        ZImpl(set_primitive_output("value", ZImpl(get_param<float>("value"))));
    }
};

ZENDEFNODE(NumericFloat, {
    {},
    {{gParamType_Float, "value"}},
    {{gParamType_Float, "value", "0"}},
    {"numeric"},
});


struct NumericVec2 : INode {
    virtual void apply() override {
        auto x = ZImpl(get_param<float>("x"));
        auto y = ZImpl(get_param<float>("y"));
        ZImpl(set_primitive_output("vec2", vec2f(x, y)));
    }
};

ZENDEFNODE(NumericVec2, {
    {},
    {{gParamType_Vec2f, "vec2"}},
    {{gParamType_Float, "x", "0"}, {gParamType_Float, "y", "0"}},
    {"numeric"},
});


struct NumericVec3 : INode {
    virtual void apply() override {
        auto x = ZImpl(get_param<float>("x"));
        auto y = ZImpl(get_param<float>("y"));
        auto z = ZImpl(get_param<float>("z"));
        ZImpl(set_primitive_output("vec3", vec3f(x, y, z)));
    }
};

ZENDEFNODE(NumericVec3, {
    {},
    {{gParamType_Vec3f, "vec3"}},
    {{gParamType_Float, "x", "0"}, {gParamType_Float, "y", "0"}, {gParamType_Float, "z", "0"}},
    {"numeric"},
});


struct NumericVec4 : INode {
    virtual void apply() override {
        auto x = ZImpl(get_param<float>("x"));
        auto y = ZImpl(get_param<float>("y"));
        auto z = ZImpl(get_param<float>("z"));
        auto w = ZImpl(get_param<float>("w"));
        ZImpl(set_primitive_output("vec4", vec4f(x, y, z, w)));
    }
};

ZENDEFNODE(NumericVec4, {
    {},
    {{gParamType_Vec4f, "vec4"}},
    {{gParamType_Float, "x", "0"}, {gParamType_Float, "y", "0"},
     {gParamType_Float, "z", "0"}, {gParamType_Float, "w", "0"}},
    {"numeric"},
});

struct PackNumericVecInt : INode {
    virtual void apply() override {
        auto _type = ZImpl(get_param<std::string>("type"));
        auto x = ZImpl(get_input2<int>("x"));
        auto y = ZImpl(get_input2<int>("y"));
        auto z = ZImpl(get_input2<int>("z"));
        auto w = ZImpl(get_input2<int>("w"));
        if (_type == "int") {
            ZImpl(set_primitive_output("veci", x));
        } else if (_type == "vec2i") {
            ZImpl(set_primitive_output("veci", vec2i(x, y)));
        } else if (_type == "vec3i") {
            ZImpl(set_primitive_output("veci", vec3i(x, y, z)));
        } else if (_type == "vec4i") {
            ZImpl(set_primitive_output("veci", vec4f(x, y, z, w)));
        }
    }
};

ZENDEFNODE(PackNumericVecInt, {
    {
        {gParamType_Int, "x", "0"},
        {gParamType_Int, "y", "0"},
        {gParamType_Int, "z", "0"},
        {gParamType_Int, "w", "0"},
    },
    {{"NumericObject","veci"}},
    {
        {"enum int vec2i vec3i vec4i", "type", "vec3i"},
    },
    {"numeric"},
});

struct PackNumericVec : INode {
    virtual void apply() override {
        auto _type = ZImpl(get_param<std::string>("type"));
        auto x = ZImpl(get_input2<float>("x"));
        auto y = ZImpl(get_input2<float>("y"));
        auto z = ZImpl(get_input2<float>("z"));
        auto w = ZImpl(get_input2<float>("w"));
        if (_type == "float") {
            ZImpl(set_primitive_output("vec", x));
        } else if (_type == "vec2f") {
            ZImpl(set_primitive_output("vec", vec2f(x, y)));
        } else if (_type == "vec3f") {
            ZImpl(set_primitive_output("vec", vec3f(x, y, z)));
        } else if (_type == "vec4f") {
            ZImpl(set_primitive_output("vec", vec4f(x, y, z, w)));
        }
    }
};

ZENDEFNODE(PackNumericVec, {
    {
        {gParamType_Float, "x", "0"},
        {gParamType_Float, "y", "0"},
        {gParamType_Float, "z", "0"},
        {gParamType_Float, "w", "0"},
    },
    {{"NumericObject","vec"}},
    {
        {"enum float vec2f vec3f vec4f", "type", "vec3f"},
    },
    {"numeric"},
});



struct TestVariantInt : INode {
    void apply() override {
        int val = m_pAdapter->get_input2<int>("intval");
    }
};

ZENDEFNODE(TestVariantInt, {
    {
        {gParamType_Int, "intval"}
    },
    {
    },
    {},
    {"numeric"}
});

struct CreateNumericObj : INode {
    void apply() override {
        std::string type = m_pAdapter->get_input2<std::string>("Numeric Type");
        std::unique_ptr<NumericObject> spNum;
        if (type == "Integer") {
            spNum = std::make_unique<NumericObject>(m_pAdapter->get_input2<int>("Integer Value"));
        }
        else if (type == "Float") {
            spNum = std::make_unique<NumericObject>(m_pAdapter->get_input2<float>("Float Value"));
        }
        else if (type == "Vector2") {
            spNum = std::make_unique<NumericObject>(m_pAdapter->get_input2<vec2f>("Vector2 Value"));
        }
        else if (type == "Vector3") {
            spNum = std::make_unique<NumericObject>(m_pAdapter->get_input2<vec3f>("Vector3 Value"));
        }
        else if (type == "Vector4") {
            spNum = std::make_unique<NumericObject>(m_pAdapter->get_input2<vec4f>("Vector4 Value"));
        }
        else {
            throw;
        }
        set_output("numericobj", std::move(spNum));
    }
};

ZENDEFNODE(CreateNumericObj, {
    {
        ParamPrimitive("Numeric Type",  gParamType_String, "Float", Combobox, std::vector<std::string>{"Vector2", "Vector3", "Vector4", "Integer", "Float"}),
        ParamPrimitive("Integer Value", gParamType_Int, 0, Lineedit),
        ParamPrimitive("Float Value", gParamType_Float, 0.f, Lineedit),
        ParamPrimitive("Vector2 Value", gParamType_Vec2f, vec2f(0,0), Vec2edit),
        ParamPrimitive("Vector3 Value", gParamType_Vec3f, vec3f(0,0,0), Vec3edit),
        ParamPrimitive("Vector4 Value", gParamType_Vec4f, vec4f(0,0,0,0), Vec4edit)
    },
    {
        {gParamType_NumericObj, "numericobj"}
    },
    {},
    {"numeric"}
});

struct Matrix4 : INode {
    void apply() override {
        auto r0 = get_input2_vec4f("r0");
        auto r1 = get_input2_vec4f("r1");
        auto r2 = get_input2_vec4f("r2");
        auto r3 = get_input2_vec4f("r3");

        glm::mat4 m(r0.x, r0.y, r0.z, r0.w,
            r1.x, r1.y, r1.z, r1.w,
            r2.x, r2.y, r2.z, r2.w,
            r3.x, r3.y, r3.z, r3.w);

        m_pAdapter->set_primitive_output("matrix", m);
    }
};

ZENDEFNODE(Matrix4, {
    {
        {gParamType_Vec4f, "r0", "0,0,0,0"},
        {gParamType_Vec4f, "r1", "0,0,0,0"},
        {gParamType_Vec4f, "r2", "0,0,0,0"},
        {gParamType_Vec4f, "r3", "0,0,0,0"},
    },
    {
        ParamPrimitive("matrix", gParamType_Matrix4, glm::mat4(1), NullControl, zeno::reflect::Any(), "", true)
        //{gParamType_Matrix4, "matrix"}
    },
    {},
    {"numeric"}
});


struct CreateListOfMatrix4 : INode {
    void apply() override {

    }
};


struct TestZVariant : INode {
    virtual void apply() override {
        zeno::reflect::Any val = m_pAdapter->get_param_result("num");
        if (false) {
            int iVal = m_pAdapter->get_input2<int>("num");
            m_pAdapter->set_output2("out", iVal);
        }
        if (false) {
            float fVal = m_pAdapter->get_input2<float>("num");
            set_output_float("out", fVal);
        }
        if (false) {
            std::string sVal = m_pAdapter->get_input2<std::string>("num");
            m_pAdapter->set_output2("out", sVal);
        }
        //m_pAdapter->set_output2("out", 3);
        m_pAdapter->set_output2("out", 4.f);
        //m_pAdapter->set_primitive_output("out", val);
    }
};

ZENDEFNODE(TestZVariant, {
    {
        {gParamType_AnyNumeric, "num"}
    },
    {
        {gParamType_AnyNumeric, "out"}
    },
    {},
    {"numeric"}
});


}
