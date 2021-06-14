#include <zen/zen.h>
#include <zen/GlobalState.h>

std::map<std::string, std::string> portalIns;
std::map<std::string, std::string> portals;

struct PortalIn : zen::INode {
    virtual void complete() override {
        auto name = get_param<std::string>("name");
        portalIns[name] = this->myname;
    }

    virtual void apply() override {
        auto name = get_param<std::string>("name");
        auto ref = get_input_ref("port");
        portals[name] = ref;
    }
};

ZENDEFNODE(PortalIn, {
    {"port"},
    {},
    {{"string", "name", "RenameMe!"}},
    {"portal"},
});

struct PortalOut : zen::INode {
    virtual void apply() override {
        auto name = get_param<std::string>("name");
        auto depnode = portalIns.at(name);
        this->sess->applyNode(depnode);
        auto ref = portals.at(name);
        set_output_ref("port", ref);
    }
};

ZENDEFNODE(PortalOut, {
    {},
    {"port"},
    {{"string", "name", "RenameMe!"}},
    {"portal"},
});
