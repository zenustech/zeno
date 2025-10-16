#include <zeno/zeno.h>
#include <zeno/core/Graph.h>
#include <zeno/types/DummyObject.h>
#include <zeno/extra/SubnetNode.h>
//#include <zeno/types/ConditionObject.h>
//#include <zeno/utils/safe_at.h>
//#include <cassert>


namespace zeno {
namespace {

struct Subnet : INode {
    virtual void apply() override {
        /*zeno::SubnetNode::apply();*/
    }
    NodeType type() const override {
        return Node_SubgraphNode;
    }
};

ZENDEFNODE(Subnet, {
    {},
    {},
    {},
    {"subgraph"},
});

struct SubInput : zeno::INode {
    void apply() override {
        //printf("!!! %s\n", typeid(*ZImpl(get_input("_IN_port"))).name());
        //set_output("port", ZImpl(get_input("_IN_port"))); 
        //set_output("hasValue", ZImpl(get_input("_IN_hasValue")));
    }

    CustomUI export_customui() const override {
        CustomUI ui;
        ParamGroup group;
        ParamTab tab;
        tab.groups.emplace_back(std::move(group));
        ui.inputPrims.emplace_back(std::move(tab));
        for (auto param : ZImpl(get_output_primitive_params())) {
            ui.outputPrims.emplace_back(param);
        }
        for (auto param : ZImpl(get_output_object_params())) {
            ui.outputObjs.emplace_back(param);
        }
        ui.uistyle.background = "#5F5F5F";
        return ui;
    }
};

ZENDEFNODE(SubInput, {
    {},
    {{gParamType_Bool, "hasValue"}},
    {},
    {"subgraph"},
});

struct SubOutput : zeno::INode {
    void apply() override {
    }

    ZENO_API zany get_default_output_object() {
        return clone_input("port");
    }

    CustomUI export_customui() const override {
        CustomUI ui;
        ParamGroup group;
        for (auto param : ZImpl(get_input_primitive_params())) {
            group.params.emplace_back(param);
        }
        ParamTab tab;
        tab.groups.emplace_back(std::move(group));
        ui.inputPrims.emplace_back(std::move(tab));
        for (auto param : ZImpl(get_input_object_params())) {
            ui.inputObjs.emplace_back(param);
        }
        ui.uistyle.background = "#5F5F5F";
        return ui;
    }
};

ZENDEFNODE(SubOutput, {
    {},
    {},
    {},
    {"subgraph"},
});

}
}
