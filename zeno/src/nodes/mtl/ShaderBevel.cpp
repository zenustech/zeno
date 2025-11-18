#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>

namespace zeno {

struct ShaderBevel : ShaderNodeClone<ShaderBevel> {

    virtual int determineType(EmissionPass *em) override {
        return TypeHint.at("vec3");
    }

    virtual void emitCode(EmissionPass *em) override {
        auto radius = get_input2<float>("radius");
        auto sample = get_input2<int>("sample");

        return em->emitCode("bevel(attrs," + std::to_string(radius) + "," + std::to_string(sample) + ")");
    }
};

ZENDEFNODE(ShaderBevel, {
    {
        {"int",   "sample", "4"},
        {"float", "radius", "0.01"},
    },
    {
        {"shader", "out"}
    },
    {
        {"enum WorldNormal", "out", "WorldNormal"}
    },
    {"shader"},
});

};