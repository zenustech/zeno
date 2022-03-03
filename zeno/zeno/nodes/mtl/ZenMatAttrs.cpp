#include <zeno/zeno.h>
#include <zeno/extra/ZenMatNode.h>
#include <zeno/types/ZenMatObject.h>
#include <zeno/utils/string.h>

namespace zeno {


struct ZenMatInputAttr : ZenMatNode {
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

ZENDEFNODE(ZenMatInputAttr, {
    {
        {"enum pos clr nrm", "attr", "pos"},
        {"enum float vec2 vec3 vec4", "type", "vec3"},
    },
    {
        {"tree", "out"},
    },
    {},
    {"zenMat"},
});


}
