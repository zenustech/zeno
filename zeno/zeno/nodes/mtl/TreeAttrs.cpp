#include <zeno/zeno.h>
#include <zeno/extra/TreeNode.h>
#include <zeno/types/TreeObject.h>
#include <zeno/utils/string.h>

namespace zeno {


struct TreeInputAttr : TreeNode {
    virtual int determineType(EmissionPass *em) override {
        auto attr = get_input2<std::string>("type");
        const char *tab[] = {"float", "vec2", "vec3", "vec4"};
        auto idx = std::find(std::begin(tab), std::end(tab), attr) - std::begin(tab);
        return idx + 1;
    }

    virtual void emitCode(EmissionPass *em) override {
        auto attr = get_input2<std::string>("attr");
        return em->emitCode("att_" + attr);
    }
};

ZENDEFNODE(TreeInputAttr, {
    {
        {"enum pos clr nrm", "attr", "pos"},
        {"enum float vec2 vec3 vec4", "type", "vec3"},
    },
    {
        {"tree", "out"},
    },
    {},
    {"tree"},
});


}
