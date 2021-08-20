#include <zeno/zeno.h>
#ifdef ZENO_VISUALIZATION
#include <zeno/extra/Visualization.h>
#endif
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
            set_output2("hasValue",
                    std::make_shared<zeno::ConditionObject>(false));
        } else {
            if (has_input2("setValue")) {
                auto obj = get_input2("setValue");
                graph->subEndpointSetValues[name] = std::move(obj);
            } else {
                auto obj = it->second();
                set_output2("getValue", std::move(obj));
            }
            set_output2("hasValue",
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
            set_output2("port", std::move(obj));
            set_output2("hasValue",
                    std::make_shared<zeno::ConditionObject>(true));

        } else if (auto it = graph->subInputPromises.find(name);
                it != graph->subInputPromises.end()) {
            auto obj = it->second();
            set_output2("port", std::move(obj));
            set_output2("hasValue",
                    std::make_shared<zeno::ConditionObject>(true));

        } else {
            set_output2("hasValue",
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
        if (has_input2("port")) {
            auto name = get_param<std::string>("name");
            auto obj = get_input2("port");
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


struct SetSubOutput : zeno::INode {
    virtual void complete() override {
        auto name = get_param<std::string>("name");
        graph->subOutputNodes[name] = myname;
        graph->finalOutputNodes.insert(myname);
    }

    virtual void apply() override {
        if (has_input2("port")) {
            auto name = get_param<std::string>("name");
            auto obj = get_input2("port");
            graph->subOutputs[name] = std::move(obj);
        }
    }
};

ZENDEFNODE(SetSubOutput, {
    {"port"},
    {},
    {{"string", "name", "output1"},
     {"string", "type", ""},
     {"string", "defl", ""}},
    {"subgraph"},
});


struct Subgraph : zeno::INode {
    virtual void apply() override {
        auto name = get_param<std::string>("name");

        auto subg = safe_at(graph->scene->graphs, name, "subgraph");
        assert(subg->scene == graph->scene);

#ifdef ZENO_VISUALIZATION
        // VIEW subnodes only if subgraph is VIEW'ed
        subg->isViewed = has_option("VIEW");
#endif

        for (auto const &[key, obj]: inputs) {
            subg->setGraphInput2(key, obj);
        }
        subg->applyGraph();

        for (auto &[key, obj]: subg->subOutputs) {
#ifdef ZENO_VISUALIZATION
            if (subg->isViewed && !subg->hasAnyView) {
                auto path = zeno::Visualization::exportPath();
                if (auto p = zeno::silent_any_cast<
                        std::shared_ptr<zeno::IObject>>(obj); p.has_value()) {
                    p.value()->dumpfile(path);
                }
                subg->hasAnyView = true;
            }
#endif
            set_output2(key, std::move(obj));
        }

        subg->subInputs.clear();
        subg->subOutputs.clear();
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
