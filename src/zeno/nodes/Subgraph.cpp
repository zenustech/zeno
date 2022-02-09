#include <zeno/zeno.h>
#include <zeno/extra/ISubgraphNode.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/utils/safe_at.h>
#include <cassert>


namespace {


struct FinalOutput : zeno::INode {
    virtual void complete() override {
        graph->finalOutputNodes.insert(myname);
    }

    virtual void apply() override {}
};

ZENDEFNODE(FinalOutput, {
    {},
    {},
    {},
    {"subgraph"},
});


#if 0
struct SubEndpoint : zeno::INode {
    virtual void complete() override {
        auto name = get_param<std::string>("name");
        graph->subEndpointNodes[name].insert(myname);
    }

    virtual void apply() override {
        auto name = get_param<std::string>("name");
        if (auto it = graph->subEndpointGetters.find(name);
                it == graph->subEndpointGetters.end()) {
            set_output("hasValue",
                    std::make_shared<zeno::ConditionObject>(false));
        } else {
            if (has_input("setValue")) {
                auto obj = get_input("setValue");
                graph->subEndpointSetValues[name] = std::move(obj);
            } else {
                auto obj = it->second();
                set_output("getValue", std::move(obj));
            }
            set_output("hasValue",
                    std::make_shared<zeno::ConditionObject>(true));
        }
    }
};

ZENDEFNODE(SubEndpoint, {
    {"setValue"},
    {"getValue", "hasValue"},
    {{"string", "name", "Cube"}},
    {"subgraph"},
});
#endif


struct SubInput : zeno::INode {
    virtual void complete() override {
        auto name = get_param<std::string>("name");
        graph->subInputNodes[name] = myname;
    }

    virtual void apply() override {
        auto name = get_param<std::string>("name");

        if (auto it = graph->subInputs.find(name);
                it != graph->subInputs.end()) {
            auto obj = it->second;
            set_output("port", std::move(obj));
            set_output("hasValue",
                    std::make_shared<zeno::ConditionObject>(true));

        } else if (auto it = graph->subInputPromises.find(name);
                it != graph->subInputPromises.end()) {
            auto obj = it->second();
            set_output("port", std::move(obj));
            set_output("hasValue",
                    std::make_shared<zeno::ConditionObject>(true));

        } else {
            set_output("hasValue",
                    std::make_shared<zeno::ConditionObject>(false));
        }
    }
};

ZENDEFNODE(SubInput, {
    {},
    {"port", "hasValue"},
    {{"string", "name", "input1"},
     {"string", "type", ""},
     {"string", "defl", ""}},
    {"subgraph"},
});


struct SubOutput : zeno::INode {
    virtual void complete() override {
        auto name = get_param<std::string>("name");
        graph->subOutputNodes[name] = myname;
        graph->finalOutputNodes.insert(myname);
    }

    virtual void apply() override {
        if (has_input("port")) {
            auto name = get_param<std::string>("name");
            auto obj = get_input("port");
            graph->subOutputs[name] = std::move(obj);
        }
    }
};

ZENDEFNODE(SubOutput, {
    {"port"},
    {},
    {{"string", "name", "output1"},
     {"string", "type", ""},
     {"string", "defl", ""}},
    {"subgraph"},
});


struct SubResult : zeno::INode {
    virtual void complete() override {
        auto name = get_param<std::string>("name");
        graph->subOutputNodes[name] = myname;
    }

    virtual void apply() override {
        if (has_input("port")) {
            auto name = get_param<std::string>("name");
            auto obj = get_input("port");
            graph->subOutputs[name] = std::move(obj);
        }
    }
};

ZENDEFNODE(SubResult, {
    {"port"},
    {},
    {{"string", "name", "output1"},
     {"string", "type", ""},
     {"string", "defl", ""}},
    {"subgraph"},
});


struct Subgraph : zeno::ISubgraphNode {  // to be deprecated
    virtual zeno::Graph *get_subgraph() override {
        auto name = get_param<std::string>("name");
        auto subg = safe_at(graph->scene->graphs, name, "subgraph");
        assert(subg->scene == graph->scene);
        return subg;
    }
};

ZENDEFNODE(Subgraph, {
    {},//"input1", "input2", "input3", "input4"},
    {},//"output1", "output2", "output3", "output4"},
    {{"string", "name", "DoNotUseThisNodeDirectly"}},
    {"subgraph"},
});


struct SubCategory : zeno::INode {
    virtual void apply() override {}
};

ZENDEFNODE(SubCategory, {
    {},
    {},
    {{"string", "name", "subgraph"}},
    {"subgraph"},
});


}
