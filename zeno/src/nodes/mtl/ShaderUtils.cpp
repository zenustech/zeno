#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>
#include "zeno/types/PrimitiveObject.h"
#include "zeno/types/UserData.h"
#include "zeno/utils/format.h"

namespace zeno {


struct ShaderLinearFit : ShaderNodeClone<ShaderLinearFit> {
    virtual int determineType(EmissionPass *em) override {
        auto in = em->determineType(ZImpl(get_input_shader("in")));
        auto inMin = em->determineType(ZImpl(get_input_shader("inMin")));
        auto inMax = em->determineType(ZImpl(get_input_shader("inMax")));
        auto outMin = em->determineType(ZImpl(get_input_shader("outMin")));
        auto outMax = em->determineType(ZImpl(get_input_shader("outMax")));

        if (inMin == 1 && inMax == 1 && outMin == 1 && outMax == 1) {
            return in;
        } else if (inMin == in && inMax == in && outMin == in && outMax == in) {
            return in;
        } else if (inMin == 1 && inMax == 1 && outMin == in && outMax == in) {
            return in;
        } else if (inMin == in && inMax == in && outMin == 1 && outMax == 1) {
            return in;
        } else {
            throw zeno::Exception("vector dimension mismatch in linear fit");
        }
    }

    virtual void emitCode(EmissionPass *em) override {
        auto in = em->determineExpr(ZImpl(get_input_shader("in")));
        auto inMin = em->determineExpr(ZImpl(get_input_shader("inMin")));
        auto inMax = em->determineExpr(ZImpl(get_input_shader("inMax")));
        auto outMin = em->determineExpr(ZImpl(get_input_shader("outMin")));
        auto outMax = em->determineExpr(ZImpl(get_input_shader("outMax")));

        auto exp = "(" + in + " - " + inMin + ") / (" + inMax + " - " + inMin + ")";
        if (ZImpl(get_param<bool>("clamped")))
            exp = "clamp(" + exp + ", 0.0, 1.0)";
        em->emitCode(exp + " * (" + outMax + " - " + outMin + ") + " + outMin);
    }
};

ZENDEFNODE(ShaderLinearFit, {
    {
        {gParamType_Float, "in", "0"},
        {gParamType_Float, "inMin", "0"},
        {gParamType_Float, "inMax", "1"},
        {gParamType_Float, "outMin", "0"},
        {gParamType_Float, "outMax", "1"},
    },
    {
        {gParamType_Float, "out"},
    },
    {
        {gParamType_Bool, "clamped", "0"},
    },
    {"shader"},
});

struct ShaderVecConvert : ShaderNodeClone<ShaderVecConvert> {
    int ty{};

    virtual int determineType(EmissionPass *em) override {
        auto _type = ZImpl(get_param<std::string>("type"));
        em->determineType(ZImpl(get_input_shader("in")));
        if (_type == "vec2") {
            ty = 2;
        }
        else if (_type == "vec3") {
            ty = 3;
        }
        else if (_type == "vec4") {
            ty = 4;
        }
        return ty;
    }

    virtual void emitCode(EmissionPass *em) override {
        std::string exp = em->determineExpr(ZImpl(get_input_shader("in")));
        em->emitCode(em->funcName("convertTo" + std::to_string(ty)) + "(" + exp + ")");
    }
};

ZENDEFNODE(ShaderVecConvert, {
    {
        { "float", "in", "0"},
    },
    {{"object", "out"}},
    {
        {"enum vec2 vec3 vec4", "type", "vec3"},
    },
    {"shader"},
});

struct ShaderVecExtract : ShaderNodeClone<ShaderVecExtract> {
    int ty{};

    virtual int determineType(EmissionPass *em) override {
        auto _type = ZImpl(get_param<std::string>("type"));
        em->determineType(ZImpl(get_input_shader("in")));
        if (_type == "xyz" || _type == "xyz(srgb)") {
            ty = 3;
        }
        else {
            ty = 1;
        }
        return ty;
    }

    virtual void emitCode(EmissionPass *em) override {
        std::string exp = em->determineExpr(ZImpl(get_input_shader("in")));
        auto _type = ZImpl(get_param<std::string>("type"));
        if (_type == "xyz") {
            em->emitCode(em->funcName("convertTo3(" + exp + ")"));
        }
        else if (_type == "xyz(srgb)") {
            em->emitCode(em->funcName("pow(convertTo3(" + exp + "), 2.2f)"));
        }
        else if (_type == "1-w"){
            em->emitCode(zeno::format("(1-{}.w)", exp));
        }
        else {
            em->emitCode(exp + "." + _type);
        }
    }
};

ZENDEFNODE(ShaderVecExtract, {
    {
        {"object", "in"},
    },
    {{"object", "out"}},
    {
        {"enum x y z w xyz 1-w xyz(srgb)", "type", "xyz"},
    },
    {"shader"},
});

struct ShaderNormalMap : ShaderNodeClone<ShaderNormalMap> {
    virtual int determineType(EmissionPass *em) override {
        auto in1 = ZImpl(get_input_shader("normalTexel"));
        auto in2 = ZImpl(get_input_shader("scale"));
        em->determineType(in1);
        em->determineType(in2);
        return 3;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto in1 = em->determineExpr(ZImpl(get_input_shader("normalTexel")));
        auto in2 = em->determineExpr(ZImpl(get_input_shader("scale")));

        return em->emitCode(em->funcName("normalmap") + "(" + in1 + ", " + in2 + ")");
    }
};

ZENDEFNODE(ShaderNormalMap, {
    {
        {gParamType_Vec3f, "normalTexel", "0.5,0.5,1.0"},
        {gParamType_Float, "scale", "1"},
    },
    {
        {gParamType_Vec3f, "out"},
    },
    {},
    {"shader"},
});

struct CalcCameraUp : INode {
    virtual void apply() override {
        auto refUp = zeno::normalize(ZImpl(get_input2<zeno::vec3f>("refUp")));
        auto pos = ZImpl(get_input2<zeno::vec3f>("pos"));
        auto target = ZImpl(get_input2<zeno::vec3f>("target"));
        vec3f view = zeno::normalize(target - pos);
        vec3f right = zeno::cross(view, refUp);
        vec3f up = zeno::cross(right, view);
        ZImpl(set_output2("pos", pos));
        ZImpl(set_output2("up", up));
        ZImpl(set_output2("view", view));
    }
};

ZENDEFNODE(CalcCameraUp, {
    {
        {gParamType_Vec3f, "refUp", "0, 1, 0"},
        {gParamType_Vec3f, "pos", "0, 0, 5"},
        {gParamType_Vec3f, "target", "0, 0, 0"},
    },
    {
        {gParamType_Vec3f, "pos"},
        {gParamType_Vec3f, "up"},
        {gParamType_Vec3f, "view"},
    },
    {},
    {"shader"},
});


struct SetPrimInvisible : INode {
    virtual void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        int invisible = ZImpl(get_input2<int>("invisible"));
        prim->userData()->set_int("invisible", invisible);

        ZImpl(set_output("out", std::move(prim)));
    }
};

ZENDEFNODE(SetPrimInvisible, {
    {
        { gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly },
        { gParamType_Bool, "invisible", "1"},
    },
    {
        {gParamType_Primitive, "out" },
    },
    {},
    { "shader" },
});
}
