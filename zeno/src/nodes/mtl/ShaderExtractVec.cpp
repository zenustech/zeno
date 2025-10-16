#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>

namespace zeno {

struct ShaderExtractVecImpl : ShaderNodeClone<ShaderExtractVecImpl> {
    virtual int determineType(EmissionPass* em) override {
        auto in1 = em->determineType(ZImpl(get_input_shader("vec")));
        return 1;
    }

    virtual void emitCode(EmissionPass* em) override {
        auto in = em->determineExpr(ZImpl(get_input_shader("vec")));
        std::string comp = ZImpl(get_input2<std::string>("comp"));
        if (comp == "x") {
            em->emitCode(in + "." + "xyzw"[0]);
        }
        else if (comp == "y") {
            em->emitCode(in + "." + "xyzw"[1]);
        }
        else if (comp == "z") {
            em->emitCode(in + "." + "xyzw"[2]);
        }
        else if (comp == "w") {
            em->emitCode(in + "." + "xyzw"[3]);
        }
    }
};

ZENDEFNODE(ShaderExtractVecImpl, {
    {
        {gParamType_Vec3f, "vec"},
    },
    {
        {gParamType_Shader, "out"}
    },
    {
        {"enum x y z w", "comp", "x"},
    },
    {"shader"},
});

#if 0
namespace {
struct ImplShaderExtractVec : ShaderNodeClone<ImplShaderExtractVec> {
    int comp{};

    virtual int determineType(EmissionPass *em) override {
        auto in1 = em->determineType(ZImpl(clone_input("vec")).get());
        return 1;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto in = em->determineExpr(ZImpl(clone_input("vec")).get());

        em->emitCode(in + "." + "xyzw"[comp]);
    }
};
}

struct ShaderExtractVec : INode {
    virtual void apply() override {
        for (int i = 0; i < 4; i++) {
            auto node = std::make_unique<ImplShaderExtractVec>();
            node->inputs["vec"] = get_input("vec");
            node->comp = i;
            auto shader = std::make_unique<ShaderObject>(node.get());
            ZImpl(set_output(std::string{} + "xyzw"[i], std::move(shader)));
        }
    }
};


ZENDEFNODE(ShaderExtractVec, {
    {
        {gParamType_Vec3f, "vec"},
    },
    {
        {gParamType_Float, "x"},
        {gParamType_Float, "y"},
        {gParamType_Float, "z"},
        {gParamType_Float, "w"},
    },
    {},
    {"shader"},
});


struct ShaderReduceVec : ShaderNodeClone<ShaderReduceVec> {
    int tyin{};

    virtual int determineType(EmissionPass *em) override {
        tyin = em->determineType(ZImpl(clone_input("in")).get());
        return 1;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto in = em->determineExpr(ZImpl(clone_input("in")).get());
        if (tyin == 1) {
            return em->emitCode(in);
        } else {
            std::string exp = in + ".x";
            for (int i = 1; i < tyin; i++) {
                exp += " + " + in + "." + "xyzw"[i];
            }
            exp = "float(" + exp + ")";
            if (ZImpl(get_param<std::string>("op")) == "average")
                exp += " / " + std::to_string(tyin) + ".";
            em->emitCode(exp);
        }
    }
};


ZENDEFNODE(ShaderReduceVec, {
    {
        {gParamType_Vec3f, "in"},
    },
    {
        {gParamType_Float, "out"},
    },
    {
        {"enum average sum", "op", "average"},
    },
    {"shader"},
});
#endif

}
