#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>

namespace zeno {


struct ShaderLinearFit : ShaderNodeClone<ShaderLinearFit> {
    virtual int determineType(EmissionPass *em) override {
        auto in = em->determineType(get_input("in").get());
        auto inMin = em->determineType(get_input("inMin").get());
        auto inMax = em->determineType(get_input("inMax").get());
        auto outMin = em->determineType(get_input("outMin").get());
        auto outMax = em->determineType(get_input("outMax").get());

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
        auto in = em->determineExpr(get_input("in").get());
        auto inMin = em->determineExpr(get_input("inMin").get());
        auto inMax = em->determineExpr(get_input("inMax").get());
        auto outMin = em->determineExpr(get_input("outMin").get());
        auto outMax = em->determineExpr(get_input("outMax").get());

        auto exp = "(" + in + " - " + inMin + ") / (" + inMax + " - " + inMin + ")";
        if (get_param<bool>("clamped"))
            exp = "clamp(" + exp + ", 0.0, 1.0)";
        em->emitCode(exp + " * (" + outMax + " - " + outMin + ") + " + outMax);
    }
};

ZENDEFNODE(ShaderLinearFit, {
    {
        {"float", "in", "0"},
        {"float", "inMin", "0"},
        {"float", "inMax", "1"},
        {"float", "outMin", "0"},
        {"float", "outMax", "1"},
    },
    {
        {"float", "out"},
    },
    {
        {"bool", "clamped", "0"},
    },
    {"shader"},
});


}
