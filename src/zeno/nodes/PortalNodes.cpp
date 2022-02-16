#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/safe_at.h>
#include <zeno/core/Graph.h>

namespace zeno {

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
        auto depnode = zeno::safe_at(graph->portalIns, name, "PortalIn");
        graph->applyNode(depnode);
        auto obj = zeno::safe_at(graph->portals, name, "portal object");
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
            log_error("requested object doesn't support clone");
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
            log_error("requested object doesn't support assign or type mismatch");
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


struct MoveClone : zeno::INode {
    virtual void apply() override {
        auto obj = get_input("object");
        auto newobj = obj->move_clone();
        if (!newobj) {
            log_error("requested object doesn't support move_clone");
            return;
        }
        set_output("newObject", std::move(newobj));
    }
};

ZENDEFNODE(MoveClone, {
    {"object"},
    {"newObject"},
    {},
    {"portal"},
});


struct MoveDelete : zeno::INode {
    virtual void apply() override {
        auto obj = get_input("object");
        auto newobj = obj->move_clone();
        if (!newobj) {
            log_error("requested object doesn't support move_clone");
            return;
        }
        newobj = nullptr;
    }
};

ZENDEFNODE(MoveDelete, {
    {"object"},
    {},
    {},
    {"portal"},
});


struct MoveAssign : zeno::INode {
    virtual void apply() override {
        auto src = get_input("src");
        auto dst = get_input("dst");
        bool succ = dst->move_assign(src.get());
        if (!succ) {
            log_error("requested object doesn't support move_assign or type mismatch");
            return;
        }
        set_output("dst", std::move(dst));
    }
};

ZENDEFNODE(MoveAssign, {
    {"dst", "src"},
    {"dst"},
    {},
    {"portal"},
});


struct SetUserData : zeno::INode {
    virtual void apply() override {
        auto object = get_input("object");
        auto key = get_param<std::string>("key");
        object->userData().set(key, get_input("data"));
        set_output("object", std::move(object));
    }
};

ZENDEFNODE(SetUserData, {
    {"object", "data"},
    {"object"},
    {{"string", "key", ""}},
    {"portal"},
});


struct GetUserData : zeno::INode {
    virtual void apply() override {
        auto object = get_input("object");
        auto key = get_param<std::string>("key");
        auto hasValue = object->userData().has(key);
        auto data = object->userData().get(key);
        set_output2("hasValue", hasValue);
        set_output("data", std::move(data));
    }
};

ZENDEFNODE(GetUserData, {
    {"object"},
    {"data", {"bool", "hasValue"}},
    {{"string", "key", ""}},
    {"portal"},
});


struct DelUserData : zeno::INode {
    virtual void apply() override {
        auto object = get_input("object");
        auto key = get_param<std::string>("key");
        object->userData().del(key);
    }
};

ZENDEFNODE(DelUserData, {
    {"object"},
    {},
    {{"string", "key", ""}},
    {"portal"},
});


struct CopyAllUserData : zeno::INode {
    virtual void apply() override {
        auto src = get_input("src");
        auto dst = get_input("dst");
        dst->userData() = src->userData();
        set_output("dst", std::move(dst));
    }
};

ZENDEFNODE(CopyAllUserData, {
    {"dst", "src"},
    {"dst"},
    {},
    {"portal"},
});


}
