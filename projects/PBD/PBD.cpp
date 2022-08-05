#include <zeno\zeno.h>
#include <zeno\types\PrimitiveObject.h>
#include "PBD.h"

namespace zeno {
    struct PBD : zeno::INode {
        virtual void apply() override 
        {
            if (has_input("prim")) 
            {
                auto prim = get_input<PrimitiveObject>("prim");
                auto &pos = prim->attr<vec3f>("pos");
                for(int i = 0; i < pos.size(); i++)
                {
                    pos[i] += 1.0;
                }
                set_output("prim", std::move(prim));
            }
        };
    };

    ZENDEFNODE(PBD, {
        // inputs:
        {"prim"},
        // outputs:
        {"prim"},
        // params:
        {{"vec3f","external_force","0, -9.8, 0"}},
        //category
        {"PBD"},
    });


} // namespace zeno
