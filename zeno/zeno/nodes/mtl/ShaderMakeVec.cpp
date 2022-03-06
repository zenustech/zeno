#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>

namespace zeno {


struct ShaderMakeVec : ShaderNode {
    int ty{};

    virtual int determineType(EmissionPass *em) override {
        auto t1 = has_input("x") ? em->determineType(get_input("x").get()) : 0;
        auto t2 = has_input("y") ? em->determineType(get_input("y").get()) : 0;
        auto t3 = has_input("z") ? em->determineType(get_input("z").get()) : 0;
        auto t4 = has_input("w") ? em->determineType(get_input("w").get()) : 0;
        ty = t1 + t2 + t3 + t4;
        if (ty > 4)
            throw zeno::Exception("ShaderMakeVec expect sum of dimension to no more than 4");
        return ty;
    }

    virtual void emitCode(EmissionPass *em) override {
        std::string exp;
        if (has_input("x")) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(get_input("x").get());
        }
        if (has_input("y")) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(get_input("y").get());
        }
        if (has_input("z")) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(get_input("z").get());
        }
        if (has_input("w")) {
            if (!exp.empty()) exp += ", ";
            exp += em->determineExpr(get_input("w").get());
        }

        em->emitCode(em->typeNameOf(ty) + "(" + exp + ")");
    }
};


ZENDEFNODE(ShaderMakeVec, {
    {
        {"shader", "x"},
        {"shader", "y"},
        {"shader", "z"},
        {"shader", "w"},
    },
    {
        {"vec4f", "out"},
    },
    {},
    {"shader"},
});


struct ShaderFillVec : ShaderNode {
    virtual int determineType(EmissionPass *em) override {
        auto tin = em->determineType(get_input("in").get());
        if (tin != 1)
            throw zeno::Exception("ShaderFillVec expect scalar as input");

        auto type = get_input2<std::string>("type");
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
        auto in = em->determineExpr(get_input("in").get());
        auto type = get_input2<std::string>("type");
        const char *tab[] = {"float", "vec2", "vec3", "vec4"};
        auto ty = std::find(std::begin(tab), std::end(tab), type) - std::begin(tab) + 1;
        em->duplicateIfHlsl(ty, in);
        em->emitCode(em->typeNameOf(ty) + "(" + in + ")");
    }
};


ZENDEFNODE(ShaderFillVec, {
    {
        {"float", "in", "0"},
        {"enum float vec2 vec3 vec4", "type", "vec3"},
    },
    {
        {"vec4f", "out"},
    },
    {},
    {"shader"},
});


}
