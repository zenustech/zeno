#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <cstdio>

// just writing some bull shit first

// struct Material : zeno::IObject {
//     // std::string vert;
//     // std::string frag;
// };

struct MakeMaterialPrimtive : zeno::INode {
    virtual void apply() override {
        printf("MakeMaterial::apply() called!\n");
        // get the input socket `prim`, assume the type is PrimitiveObject
        std::shared_ptr<zeno::PrimitiveObject> prim = get_input<zeno::PrimitiveObject>("prim");
        
        // create a new attribute called `mtr`
        // Need to define a new type in types (MaterialObject.h?)
        // <> &mtr = prim->add_attr<>("mtr");
        
        set_output("prim", std::move(prim));
    }
};

// after that, invoke ZENDEFNODE to actually load the node into ZENO:
ZENDEFNODE(
    MakeMaterialPrimtive,
    { 
        /* inputs: */ 
        {
            {"primitive", "prim"},
        }, 
        /* outputs: */ 
        {
            {"primitive", "prim"},
        }, 
        /* params: */ 
        {}, 
        /* category: */ 
        {
            "Material",
        }
    }
);