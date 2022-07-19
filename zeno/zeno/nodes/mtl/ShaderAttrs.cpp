#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>

namespace zeno {


struct ShaderInputAttr : ShaderNodeClone<ShaderInputAttr> {
    virtual int determineType(EmissionPass *em) override {
        auto type = get_input2<std::string>("type");
        const char *tab[] = {"float", "vec2", "vec3", "vec4"};
        auto idx = std::find(std::begin(tab), std::end(tab), type) - std::begin(tab);
        return idx + 1;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto attr = get_input2<std::string>("attr");
        return em->emitCode("att_" + attr);
    }
};

ZENDEFNODE(ShaderInputAttr, {
    {
        {"enum pos clr nrm uv tang bitang NoL", "attr", "pos"},
        {"enum float vec2 vec3 vec4", "type", "vec3"},
    },
    {
        {"shader", "out"},
    },
    {},
    {"shader"},
});

struct ShaderInputUniform : ShaderNodeClone<ShaderInputUniform> {
    virtual int determineType(EmissionPass *em) override {
        auto type = get_input2<std::string>("type");
        const char *tab[] = {"float", "vec2", "vec3", "vec4"};
        auto idx = std::find(std::begin(tab), std::end(tab), type) - std::begin(tab);
        return idx + 1;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto index = get_input2<int>("index");
        auto type = get_input2<std::string>("type");
        return em->emitCode("attr_uniform_" + type + "[" + std::to_string(index) + "]");
    }
};

ZENDEFNODE(ShaderInputUniform, {
    {
        {"enum float vec2 vec3 vec4", "type", "vec3"},
        {"int", "index", "0"}
    },
    {
        {"shader", "out"},
    },
    {},
    {"shader"},
});

}
