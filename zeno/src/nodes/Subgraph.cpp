#include <zeno/zeno.h>
#include <zeno/core/Graph.h>
#include <zeno/types/DummyObject.h>
#include <zeno/extra/SubnetNode.h>
//#include <zeno/types/ConditionObject.h>
//#include <zeno/utils/safe_at.h>
//#include <cassert>


namespace zeno {
namespace {

struct Subnet : zeno::SubnetNode {
    virtual void apply() override {
        zeno::SubnetNode::apply();
    }
};

ZENDEFNODE(Subnet, {
    {},
    {},
    {},
    {"subgraph"},
});

struct SubInput : zeno::INode {
    virtual void complete() override {
        auto name = get_param<std::string>("name");
        //graph->subInputNodes[name] = myname;
    }

    virtual void apply() override {
        //直接由SubnetNode来设置output，这里其实不用再apply了。
        //printf("!!! %s\n", typeid(*get_input("_IN_port")).name());
        //set_output("port", get_input("_IN_port")); 
        //set_output("hasValue", get_input("_IN_hasValue"));
    }
};

ZENDEFNODE(SubInput, {
    {},
    {{"bool", "hasValue"}},
    {},
    {"subgraph"},
});

struct SubOutput : zeno::INode {
    virtual void complete() override {
    }

    virtual void apply() override {
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
