#include <zen/zen.h>
#include <zen/GlobalState.h>

static std::map<std::string, std::string> portalIns;
static std::map<std::string, std::shared_ptr<zen::IObject>> portals;

struct PortalIn : zen::INode {
    virtual void complete() override {
        auto name = get_param<std::string>("name");
        portalIns[name] = this->myname;
    }

    virtual void apply() override {
        auto name = get_param<std::string>("name");
        auto obj = get_input("port");
        portals[name] = std::move(obj);
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
        sess->applyNode(depnode);
        auto obj = portals.at(name);
        set_output("port", std::move(obj));
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
        auto obj = get_input("input");
        set_output("output", std::move(obj));
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

