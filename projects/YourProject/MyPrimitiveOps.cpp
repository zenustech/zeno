// An demo on how to work with the built-in PrimitiveObject of ZENO
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <cstdio>

// a simple node that colorize a PrimitiveObject by XYZ coordinates of vertices
// all nodes should be a derived class of zeno::INode
struct MyColorizePrimitive : zeno::INode {
    // the virtual function apply() is called whenever ZENO execute it
    // now let's override it to define our own operations
    virtual void apply() override {
        printf("MyColorizePrimitive::apply() called!\n");

        // get the input socket `prim`, assume the type is PrimitiveObject
        std::shared_ptr<zeno::PrimitiveObject> prim = get_input<zeno::PrimitiveObject>("prim");

        // create a new attribute called `clr`
        std::vector<zeno::vec3f> &clr = prim->add_attr<zeno::vec3f>("clr");

        // get the existing attribute called `pos`
        std::vector<zeno::vec3f> &pos = prim->attr<zeno::vec3f>("pos");

        // it is ensured that: prim->size() == clr.size() == pos.size()
        for (int i = 0; i < prim->size(); i++) {
            // pos[i] is the position of the i-th point
            // clr[i] is the color of the i-th point
            // calculating `clr` from `pos` attribute for each point i
            clr[i] = pos[i] * 0.5 + 0.5;
        }

        // the visualizer will automatically shade primitive by looking up
        // the `clr` attribute of it. Also for `nrm` which is the normal
        // vector of mesh primitives, and `rad` for the radius of particle
        // primitives.

        // set the output to reference the same pointer as input `prim`.
        // later nodes is ensured to be executed later if connected to
        // this output `prim`, using `SRC` and `DST` would works too.
        set_output("prim", std::move(prim));
    }
};

// after that, invoke ZENDEFNODE to actually load the node into ZENO:
ZENDEFNODE(
    MyColorizePrimitive,
    { 
        /* inputs: */ 
        {
            // input descriptors follows the format: {type, name}
            // the `name` is that you can use in get_input(name)
            // the `type` is just a hint to the editor to prevent
            // user from misconnecting wrong type to them
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
            "YourProject",
        }
    }
);


// a simple node that translate a PrimitiveObject by a given offset
struct MyTranslatePrimitive : zeno::INode {
    virtual void apply() override {
        printf("MyTranslatePrimitive::apply() called!\n");
        
        // get the input socket `prim`, assume the type is PrimitiveObject
        std::shared_ptr<zeno::PrimitiveObject> prim = get_input<zeno::PrimitiveObject>("prim");
        // get the input socket `offset`, assume the type is NumericObject
        std::shared_ptr<zeno::NumericObject> offset_obj = get_input<zeno::NumericObject>("offset");


        // get the actual offset vector from NumericObject (assumed NumericVec3)
        zeno::vec3f offset = offset_obj->get<zeno::vec3f>();
        // alternatively use get<float>() for scalars (assumed NumericFloat)

        // access the zeno::vec3f components using offset[0] rather than offset.x
        printf("translating primitive with offset: %f %f %f\n", offset[0], offset[1], offset[2]);

        // get the existing attribute called `pos` 
        
        // important (don't miss the & symbol!)
        std::vector<zeno::vec3f> &pos = prim->attr<zeno::vec3f>("pos");

        // it is ensured that: prim->size() == pos.size()
        for (int i = 0; i < prim->size(); i++) {
            // translate all vertices by a uniform offset
            pos[i] = pos[i] + offset;
        }

        // always wise to use std::move..
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(
    MyTranslatePrimitive,
    { 
        /* inputs: */ 
        {
            {"primitive", "prim"},
            {"numeric:vec3f", "offset"},
        }, 
        /* outputs: */ 
        {
            {"primitive", "prim"},
        }, 
        /* params: */ 
        {}, 
        /* category: */ 
        {
            "YourProject",
        }
    }
);
