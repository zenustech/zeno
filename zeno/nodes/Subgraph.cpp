#include <zeno/zeno.h>
#include <zeno/GlobalState.h>
#ifdef ZENO_VISUALIZATION
#include <zeno/Visualization.h>
#endif
#include <zeno/ConditionObject.h>
#include <zeno/safe_at.h>
#include <cassert>


namespace {

struct SubInput : zeno::INode {
    virtual void apply() override {
        auto name = get_param<std::string>("name");
        if (auto it = graph->subInputs.find(name);
                it == graph->subInputs.end()) {
            set_output("hasValue",
                    std::make_shared<zeno::ConditionObject>(false));
        } else {
            auto obj = it->second;
            set_output("port", std::move(obj));
            set_output("hasValue",
                    std::make_shared<zeno::ConditionObject>(true));
        }
    }
};

ZENDEFNODE(SubInput, {
    {},
    {"port", "hasValue"},
    {{"string", "type", ""},
     {"string", "name", "input1"},
     {"string", "defl", ""}},
    {"subgraph"},
});


struct SubOutput : zeno::INode {
    virtual void apply() override {
        auto name = get_param<std::string>("name");
        auto obj = get_input("port");
        graph->subOutputs[name] = std::move(obj);
    }
};

ZENDEFNODE(SubOutput, {
    {"port"},
    {},
    {{"string", "type", ""},
     {"string", "name", "output1"},
     {"string", "defl", ""}},
    {"subgraph"},
});


struct Subgraph : zeno::INode {
    virtual void apply() override {
        auto name = get_param<std::string>("name");

        auto subg = graph->sess->graphs.at(name).get();
        assert(subg->sess == graph->sess);

#ifdef ZENO_VISUALIZATION
        // VIEW subnodes only if subgraph is VIEW'ed
        subg->isViewed = has_option("VIEW");
#endif

        for (auto const &[key, obj]: inputs) {
            subg->subInputs[key] = obj;
        }

        std::vector<std::string> applies;
        for (auto const &[key, node]: subg->nodes) {
            auto suboutptr = dynamic_cast<SubOutput *>(node.get());
            if (suboutptr) {
                applies.push_back(key);
            }
        }
        subg->applyNodes(applies);

#ifdef ZENO_VISUALIZATION
        for (auto &[key, obj]: subg->subOutputs) {
            if (subg->isViewed && !subg->hasAnyView) {
                auto path = zeno::Visualization::exportPath();
                obj->dumpfile(path);
                subg->hasAnyView = true;
            }
            set_output(key, std::move(obj));
        }
#endif

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
