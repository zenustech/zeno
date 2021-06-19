#include <zen/zen.h>
#include <zen/GlobalState.h>

static std::map<std::string, std::string> portalIns;
static std::map<std::string, std::string> portals;

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


struct Route : zen::INode {
    virtual void apply() override {
        auto ref = get_input_ref("input");
        set_output_ref("output", ref);
    }
};

ZENDEFNODE(Route, {
    {"input"},
    {"output"},
    {},
    {"portal"},
});


struct Clone : zen::INode {
    virtual void apply() override {
        auto obj = get_input("object");
        auto newobj = obj->clone();
        if (!newobj) {
            printf("ERROR: requested object doesn't support clone\n");
            return;
        }
        set_output("newObject", std::move(newobj));
    }
};

ZENDEFNODE(Clone, {
    {"object"},
    {"newObject"},
    {},
    {"portal"},
});

