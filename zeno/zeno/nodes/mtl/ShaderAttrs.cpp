#include <zeno/zeno.h>
#include <zeno/extra/ShaderNode.h>
#include <zeno/types/ShaderObject.h>
#include <zeno/utils/string.h>

namespace zeno {


struct ShaderInputAttr : ShaderNode {
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
        {"enum pos clr nrm uv", "attr", "pos"},
        {"enum float vec2 vec3 vec4", "type", "vec3"},
    },
    {
        {"tree", "out"},
    },
    {},
    {"shader"},
});


}
