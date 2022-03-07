#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>

namespace zeno {


namespace {
struct ImplShaderExtractVec : ShaderNode {
    int comp{};

    virtual int determineType(EmissionPass *em) override {
        auto in1 = em->determineType(get_input("vec").get());
        return 1;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto in = em->determineExpr(get_input("vec").get());

        em->emitCode(in + "." + "xyzw"[comp]);
    }
};
}

struct ShaderExtractVec : INode {
    virtual void apply() override {
        for (int i = 0; i < 4; i++) {
            auto node = std::make_shared<ImplShaderExtractVec>();
            node->inputs["vec"] = get_input("vec");
            node->comp = i;
            auto shader = std::make_shared<ShaderObject>(node.get());
            shader->extra_data = std::move(node);
            set_output(std::string{} + "xyzw"[i], std::move(shader));
        }
    }
};


ZENDEFNODE(ShaderExtractVec, {
    {
        {"vec3f", "vec"},
    },
    {
        {"float", "x"},
        {"float", "y"},
        {"float", "z"},
        {"float", "w"},
    },
    {},
    {"shader"},
});


struct ShaderReduceVec : ShaderNode {
    int tyin{};

    virtual int determineType(EmissionPass *em) override {
        tyin = em->determineType(get_input("in").get());
        return 1;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto in = em->determineExpr(get_input("in").get());
        if (tyin == 1) {
            return em->emitCode(in);
        } else {
            std::string exp = in + ".x";
            for (int i = 1; i < tyin; i++) {
                exp += " + " + in + "." + "xyzw"[i];
            }
            exp = "float(" + exp + ")";
            if (get_param<std::string>("op") == "average")
                exp += " / " + std::to_string(tyin) + ".";
            em->emitCode(exp);
        }
    }
};


ZENDEFNODE(ShaderReduceVec, {
    {
        {"vec3f", "in"},
    },
    {
        {"float", "out"},
    },
    {
        {"enum average sum", "op", "average"},
    },
    {"shader"},
});


}
