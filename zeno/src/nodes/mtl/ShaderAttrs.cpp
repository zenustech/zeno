#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>
#include <algorithm>

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

struct ShaderUniformAttr : ShaderNodeClone<ShaderUniformAttr> {
    virtual int determineType(EmissionPass *em) override {
        auto type = get_input2<std::string>("type");
        const char *tab[] = {"float", "vec2", "vec3", "vec4"};
        auto idx = std::find(std::begin(tab), std::end(tab), type) - std::begin(tab);
        return idx + 1;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto idx = get_input2<int>("idx");
        auto type = get_input2<std::string>("type");
        return em->emitCode(type + "(vec4(uniforms[" + std::to_string(idx) + "]))");
    }
};

ZENDEFNODE(ShaderUniformAttr, {
                                {
                                    {"int", "idx", "0"},
                                    {"enum float vec2 vec3 vec4", "type", "vec3"},
                                },
                                {
                                    {"shader", "out"},
                                },
                                {},
                                {"shader"},
                            });

}
