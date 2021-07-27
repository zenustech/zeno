#include <zeno/zeno.h>
#include <zeno/GlobalState.h>
#include <zeno/ConditionObject.h>

struct PortalIn : zeno::INode {
    virtual void complete() override {
        auto name = get_param<std::string>("name");
        graph->portalIns[name] = this->myname;
    }

    virtual void apply() override {
        auto name = get_param<std::string>("name");
        auto obj = get_input("port");
        graph->portals[name] = std::move(obj);
    }
};

ZENDEFNODE(PortalIn, {
    {"port"},
    {},
    {{"string", "name", "RenameMe!"}},
    {"portal"},
});

struct PortalOut : zeno::INode {
    virtual void apply() override {
        auto name = get_param<std::string>("name");
        auto depnode = graph->portalIns.at(name);
        graph->applyNode(depnode);
        auto obj = graph->portals.at(name);
        set_output("port", std::move(obj));
    }
};

ZENDEFNODE(PortalOut, {
    {},
    {"port"},
    {{"string", "name", "RenameMe!"}},
    {"portal"},
});


struct Route : zeno::INode {
    virtual void apply() override {
        if (has_input("input")) {
            auto obj = get_input("input");
            set_output("output", std::move(obj));
        } else {
            set_output("output", std::make_shared<zeno::ConditionObject>());
        }
    }
};

ZENDEFNODE(Route, {
    {"input"},
    {"output"},
    {},
    {"portal"},
});


struct Clone : zeno::INode {
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


struct Assign : zeno::INode {
    virtual void apply() override {
        auto src = get_input("src");
        auto dst = get_input("dst");
        bool succ = dst->assign(src.get());
        if (!succ) {
            printf("ERROR: requested object doesn't support assign or type mismatch\n");
            return;
        }
        set_output("dst", std::move(dst));
    }
};

ZENDEFNODE(Assign, {
    {"dst", "src"},
    {"dst"},
    {},
    {"portal"},
});
