#include <zeno/zeno.h>
#include <zeno/core/NodeImpl.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>
#include "zeno/utils/format.h"

namespace zeno {


static const char /* see https://docs.gl/sl4/trunc */
    unops[] = "copy neg abs sqrt inversesqrt exp log sin cos tan asin acos atan degrees"
              " radians sinh cosh tanh asinh acosh atanh round roundEven floor"
              " ceil trunc sign length normalize hsvToRgb rgbToHsv luminance saturate",
    binops[] = "add sub mul div mod pow atan2 min max dot cross distance safepower step",
    ternops[] = "mix clamp smoothstep add3 ?";


static auto &toHlsl() {
    static std::map<std::string, std::string> tab;
    return tab;
}


struct ShaderTernaryMath : ShaderNodeClone<ShaderTernaryMath> {
    virtual int determineType(EmissionPass *em) override {
        auto op = ZImpl(get_input2<std::string>("op"));
        auto in1 = ZImpl(get_input_shader("in1"));
        auto in2 = ZImpl(get_input_shader("in2"));
        auto in3 = ZImpl(get_input_shader("in3"));
        auto t1 = em->determineType(in1);
        auto t2 = em->determineType(in2);
        auto t3 = em->determineType(in3);

        if (t1 == 1 && t2 == t3) {
            return t2;
        } else if (t2 == 1 && t3 == t1) {
            return t3;
        } else if (t3 == 1 && t1 == t2) {
            return t1;
        } else if (t1 == 1 && t2 == 1) {
            return t3;
        } else if (t2 == 1 && t3 == 1) {
            return t2;
        } else if (t3 == 1 && t1 == 1) {
            return t2;
        } else if (t1 == t2 && t2 == t3) {
            return t1;
        } else {
            throw zeno::Exception("vector dimension mismatch: " + std::to_string(t1) + ", " + std::to_string(t2) + ", " + std::to_string(t3));
        }
    }

    virtual void emitCode(EmissionPass *em) override {
        auto op = ZImpl(get_input2<std::string>("op"));
        auto in1 = em->determineExpr(ZImpl(get_input_shader("in1")), this);
        auto in2 = em->determineExpr(ZImpl(get_input_shader("in2")), this);
        auto in3 = em->determineExpr(ZImpl(get_input_shader("in3")), this);

        if (op == "add3") {
            return em->emitCode(in1 + " + " + in2 + " + " + in3);
        } 
        else if (op == "?") {
            return em->emitCode(in1 + " ? " + in2 + " : " + in3);
        } 
        else {
            return em->emitCode(em->funcName(op) + "(" + in1 + ", " + in2 + ", " + in3 + ")");
        }
    }
};

ZENDEFNODE(ShaderTernaryMath, {
    {
        {gParamType_Float, "in1", "0"},
        {gParamType_Float, "in2", "0"},
        {gParamType_Float, "in3", "0"},
        {(std::string)"enum " + ternops, "op", "mix"},
    },
    {
        {gParamType_Shader, "out"},
    },
    {},
    {"shader"},
});


struct ShaderBinaryMath : ShaderNodeClone<ShaderBinaryMath> {
    virtual int determineType(EmissionPass *em) override {
        auto op = ZImpl(get_input2<std::string>("op"));
        auto in1 = ZImpl(get_input_shader("in1"));
        auto in2 = ZImpl(get_input_shader("in2"));
        auto t1 = em->determineType(in1);
        auto t2 = em->determineType(in2);

        if (op == "dot") {
            if (t1 != t2)
                throw zeno::Exception("both-side of dot must have same dimension");
            else if (t1 == 1)
                throw zeno::Exception("dot only work for vectors");
            else
                return 1;

        } else if (op == "cross") {
            if (t1 != t2)
                throw zeno::Exception("both-side of cross must have same dimension");
            else if (t1 == 2)
                return 1;
            else if (t1 == 3)
                return 3;
            else
                throw zeno::Exception("dot only work for 2d and 3d vectors");

        } else if (op == "distance") {
            if (t1 != t2)
                throw zeno::Exception("both-side of distance must have same dimension");
            else if (t1 == 1)
                throw zeno::Exception("distance only work for vectors");
            else
                return t1;

        } else if (t1 == 1) {
            return t2;
        } else if (t2 == 1) {
            return t1;
        } else if (t1 == t2) {
            return t1;
        } else {
            throw zeno::Exception("vector dimension mismatch: " + std::to_string(t1) + " != " + std::to_string(t2));
        }
    }

    virtual void emitCode(EmissionPass *em) override {
        auto op = ZImpl(get_input2<std::string>("op"));
        auto in1 = em->determineExpr(ZImpl(get_input_shader("in1")));
        auto in2 = em->determineExpr(ZImpl(get_input_shader("in2")));

        if (op == "add") {
            return em->emitCode(in1 + " + " + in2);
        } else if (op == "sub") {
            return em->emitCode(in1 + " - " + in2);
        } else if (op == "mul") {
            return em->emitCode(in1 + " * " + in2);
        } else if (op == "div") {
            return em->emitCode(in1 + " / " + in2);
        } else {
            return em->emitCode(em->funcName(op) + "(" + in1 + ", " + in2 + ")");
        }
    }
};

ZENDEFNODE(ShaderBinaryMath, {
    {
        {gParamType_AnyNumeric, "in1", "0.0"},
        {gParamType_AnyNumeric, "in2", "0.0"},
        {(std::string)"enum " + binops, "op", "add"},
    },
    {
        {gParamType_Shader, "out"},
    },
    {},
    {"shader"},
});


struct ShaderUnaryMath : ShaderNodeClone<ShaderUnaryMath> {
    virtual int determineType(EmissionPass *em) override {
        auto op = ZImpl(get_input2<std::string>("op"));
        auto in1 = ZImpl(get_input_shader("in1"));
        auto t1 = em->determineType(in1);
        if(op=="length")
        {
            t1 = 1;
        }
        return t1;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto op = ZImpl(get_input2<std::string>("op"));
        auto in1 = em->determineExpr(ZImpl(get_input_shader("in1")));

        if (op == "copy") {
            return em->emitCode(in1);
        } else if (op == "neg") {
            return em->emitCode("-" + in1);
        } else {
            return em->emitCode(em->funcName(op) + "(" + in1 + ")");
        }
    }
};

ZENDEFNODE(ShaderUnaryMath, {
    {
        {gParamType_Float, "in1", "0"},
        {(std::string)"enum " + unops, "op", "sqrt"},
    },
    {
        {gParamType_Shader, "out"},
    },
    {},
    {"shader"},
});

struct ShaderHsvAdjust : ShaderNodeClone<ShaderHsvAdjust> {
    virtual int determineType(EmissionPass *em) override {
        em->determineType(ZImpl(get_input_shader("color")));
        em->determineType(ZImpl(get_input_shader("amount")));
        return 3;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto color = em->determineExpr(ZImpl(get_input_shader("color")));
        auto amount = em->determineExpr(ZImpl(get_input_shader("amount")));

        return em->emitCode(zeno::format("{}({}, {})", em->funcName("hsvAdjust"), color, amount));
    }
};

ZENDEFNODE(ShaderHsvAdjust, {
    {
        {gParamType_Vec3f, "color"},
        {gParamType_Vec3f, "amount", "0,1,1"},
    },
    {
        {gParamType_Shader, "out"},
    },
    {},
    {"shader"},
});

}
