#include <zeno/zeno.h>
#include <zeno/types/BlenderMesh.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/safe_at.h>

namespace {



struct BlenderText : zeno::INode {
    virtual void apply() override {
        auto text = get_input2<std::string>("text");
        set_output2("value", std::move(text));
    }
};

ZENDEFNODE(BlenderText, {
    {{"string", "text", "DontUseThisNodeDirectly"}},
    {{"string", "value"}},
    {},
    {"blender"},
});


struct BlenderInput : zeno::INode {
    virtual void apply() override {
        auto &inputs = graph->getUserData().get<zeno::BlenderInputsType>("blender_inputs");
        auto objid = get_input2<std::string>("objid");
        auto object = zeno::safe_at(inputs, objid, "blender input")();
        set_output2("object", std::move(object));
    }
};

ZENDEFNODE(BlenderInput, {
    {{"string", "objid", "DontUseThisNodeDirectly"}},
    {{"BlenderAxis", "object"}},
    {},
    {"blender"},
});


struct BlenderOutput : zeno::INode {
    virtual void complete() override {
        if (get_param<bool>("active")) {
            graph->finalOutputNodes.insert(myname);
        }
    }

    virtual void apply() override {
        auto &outputs = graph->getUserData().get<zeno::BlenderOutputsType>("blender_outputs");
        auto objid = get_input2<std::string>("objid");
        auto object = get_input<zeno::BlenderAxis>("object");
        outputs[objid] = std::move(object);
    }
};

ZENDEFNODE(BlenderOutput, {
    {{"string", "objid", "DontUseThisNodeDirectly"}, {"BlenderAxis", "object"}},
    {},
    {{"bool", "active", "1"}},
    {"blender"},
});

}
