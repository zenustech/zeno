#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>
#include <algorithm>

namespace zeno {

struct ShaderPackVector : ShaderNodeClone<ShaderPackVector> {
    int ty{};

    virtual int determineType(EmissionPass *em) override {
        auto _type = ZImpl(get_param<std::string>("type"));
        if (_type == "float") {
            em->determineType(ZImpl(get_input_shader("x")));
            ty = 1;
        }
        else if (_type == "vec2") {
            em->determineType(ZImpl(get_input_shader("x")));
            em->determineType(ZImpl(get_input_shader("y")));
            ty = 2;
        }
        else if (_type == "vec3") {
            em->determineType(ZImpl(get_input_shader("x")));
            em->determineType(ZImpl(get_input_shader("y")));
            em->determineType(ZImpl(get_input_shader("z")));
            ty = 3;
        }
        else if (_type == "vec4") {
            em->determineType(ZImpl(get_input_shader("x")));
            em->determineType(ZImpl(get_input_shader("y")));
            em->determineType(ZImpl(get_input_shader("z")));
            em->determineType(ZImpl(get_input_shader("w")));
            ty = 4;
        }
        return ty;
    }

    virtual void emitCode(EmissionPass *em) override {
        std::string exp;
        if (ty >= 1) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(ZImpl(get_input_shader("x")));
        }
        if (ty >= 2) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(ZImpl(get_input_shader("y")));
        }
        if (ty >= 3) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(ZImpl(get_input_shader("z")));
        }
        if (ty >= 4) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(ZImpl(get_input_shader("w")));
        }

        em->emitCode(em->typeNameOf(ty) + "(" + exp + ")");
    }
};

ZENDEFNODE(ShaderPackVector, {
    {
        {gParamType_Float, "x", "0"},
        {gParamType_Float, "y", "0"},
        {gParamType_Float, "z", "0"},
        {gParamType_Float, "w", "0"},
    },
    {{gParamType_Shader,"out"}},
    {
        {"enum float vec2 vec3 vec4", "type", "vec3"},
    },
    {"shader"},
});

struct ShaderPackVec : ShaderNodeClone<ShaderPackVec> {
    int ty{};

    virtual int determineType(EmissionPass *em) override {
        auto t1 = ZImpl(has_input("x")) ? em->determineType(ZImpl(get_input_shader("x"))) : 0;
        auto t2 = ZImpl(has_input("y")) ? em->determineType(ZImpl(get_input_shader("y"))) : 0;
        auto t3 = ZImpl(has_input("z")) ? em->determineType(ZImpl(get_input_shader("z"))) : 0;
        auto t4 = ZImpl(has_input("w")) ? em->determineType(ZImpl(get_input_shader("w"))) : 0;
        ty = t1 + t2 + t3 + t4;
        if (ty > 4)
            throw zeno::Exception("ShaderPackVec expect sum of dimension to no more than 4");
        return ty;
    }

    virtual void emitCode(EmissionPass *em) override {
        std::string exp;
        if (ZImpl(has_input("x"))) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(ZImpl(get_input_shader("x")));
        }
        if (ZImpl(has_input("y"))) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(ZImpl(get_input_shader("y")));
        }
        if (ZImpl(has_input("z"))) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(ZImpl(get_input_shader("z")));
        }
        if (ZImpl(has_input("w"))) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(ZImpl(get_input_shader("w")));
        }

        em->emitCode(em->typeNameOf(ty) + "(" + exp + ")");
    }
};


ZENDEFNODE(ShaderPackVec, {
    {
        {"shader", "x"},
        {"shader", "y"},
        {"shader", "z"},
        {"shader", "w"},
    },
    {
        {gParamType_Vec4f, "out"},
    },
    {},
    {"deprecated"},
});


struct ShaderMakeVec : ShaderPackVec {
};

ZENDEFNODE(ShaderMakeVec, {
    {
        {gParamType_Float, "x"},
        {gParamType_Float, "y"},
        {gParamType_Float, "z"},
        {gParamType_Float, "w"},
    },
    {
        {gParamType_Vec4f, "out"},
    },
    {},
    {"deprecated"},
});


struct ShaderFillVec : ShaderNodeClone<ShaderFillVec> {
    virtual int determineType(EmissionPass *em) override {
        auto tin = em->determineType(ZImpl(get_input_shader("in")));
        if (tin != 1)
            throw zeno::Exception("ShaderFillVec expect scalar as input");

        auto type = ZImpl(get_input2<std::string>("type"));
        if (type == "float")
            return 1;
        else if (type == "vec2")
            return 2;
        else if (type == "vec3")
            return 3;
        else if (type == "vec4")
            return 4;
        else
            throw zeno::Exception("ShaderFillVec got bad type: " + type);
    }

    virtual void emitCode(EmissionPass *em) override {
        auto in = em->determineExpr(ZImpl(get_input_shader("in")));
        auto type = ZImpl(get_input2<std::string>("type"));
        const char *tab[] = {"float", "vec2", "vec3", "vec4"};
        auto ty = std::find(std::begin(tab), std::end(tab), type) - std::begin(tab) + 1;
        em->duplicateIfHlsl(ty, in);
        em->emitCode(em->typeNameOf(ty) + "(" + in + ")");
    }
};


ZENDEFNODE(ShaderFillVec, {
    {
        {gParamType_Float, "in", "0"},
        {"enum float vec2 vec3 vec4", "type", "vec3"},
    },
    {
        {gParamType_Vec4f, "out"},
    },
    {},
    {"shader"},
});


}
