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
        auto &ud = graph->getUserData();
        using UDType = std::shared_ptr<std::map<std::string,
              std::function<std::shared_ptr<zeno::BlenderAxis>()>
                  >>;
        auto &inputs = *ud.get<UDType>("blender_inputs");
        auto objid = get_input2<std::string>("object");
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
    virtual void apply() override {
        auto &ud = graph->getUserData();
        using UDType = std::shared_ptr<std::map<std::string, std::shared_ptr<zeno::BlenderAxis>>>;
        auto &outputs = *ud.get<UDType>("blender_outputs");
        auto objid = get_input2<std::string>("objid");
        auto object = get_input<zeno::BlenderAxis>("object");
        outputs[objid] = std::move(object);
    }
};

ZENDEFNODE(BlenderOutput, {
    {{"string", "objid", "DontUseThisNodeDirectly"}, {"BlenderAxis", "object"}},
    {},
    {},
    {"blender"},
});

}
