#include <zeno/zeno.h>
#include "plugindef.h"

namespace zeno
{
    struct CustomPlugin1Node : zeno::INode {
        virtual void apply() override {
            auto path = get_input<StringObject>("path")->get(); // std::string
            //auto result = zeno::readWav(path);
            //set_output("prim",result);
        }
    };

    ZENDEFNODE(CustomPlugin1Node, {
        {
            {gParamType_String, "path", "", zeno::Socket_Primitve, zeno::ReadPathEdit},
        },
        {
            {gParamType_Primitive, "prim"},
        },
        {},
        {
            "customplugin1"
        },
    });

}
