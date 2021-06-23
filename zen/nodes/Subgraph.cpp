#include <zen/zen.h>
#include <zen/GlobalState.h>


struct SubInput : zen::INode {
    virtual void apply() override {
        auto name = get_param<std::string>("name");
        auto obj = graph->subInputs.at(name);
        set_output("port", std::move(obj));
    }
};

ZENDEFNODE(SubInput, {
    {},
    {"port"},
    {{"string", "name", "RenameMe!"}},
    {"subgraph"},
});


struct SubOutput : zen::INode {
    virtual void apply() override {
        auto name = get_param<std::string>("name");
        auto obj = get_input("port");
        graph->subOutputs[name] = std::move(obj);
    }
};

ZENDEFNODE(SubOutput, {
    {"port"},
    {},
    {{"string", "name", "RenameMe!"}},
    {"subgraph"},
});


struct Subgraph : zen::INode {
    virtual void apply() override {
        auto name = get_param<std::string>("name");

        auto subg = std::make_unique<zen::Graph>();
        subg->sess = graph->sess;

        std::vector<std::string> applies;
        for (auto const &[key, node]: subg->nodes) {
            auto suboutptr = dynamic_cast<SubOutput *>(node.get());
            if (suboutptr) {
                applies.push_back(key);
            }
        }
        subg->applyNodes(applies);
    }
};

ZENDEFNODE(Subgraph, {
    {"input1"},
    {"input2"},
    {{"string", "name", "RenameMe!"}},
    {"subgraph"},
});
