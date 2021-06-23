#include <zen/zen.h>
#include <zen/GlobalState.h>

static std::map<std::string, std::string> portalIns;
static std::map<std::string, std::shared_ptr<zen::IObject>> portals;

struct Subgraph : zen::INode {
    virtual void apply() override {
        auto name = get_param<std::string>("name");
        auto obj = get_input("port");
        portals[name] = std::move(obj);
    }
};

ZENDEFNODE(Subgraph, {
    {},
    {},
    {{"string", "name", "RenameMe!"}},
    {"subgraph"},
});
