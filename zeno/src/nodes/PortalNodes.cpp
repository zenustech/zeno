#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/DummyObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/safe_at.h>
#include <zeno/core/Graph.h>

namespace zeno {

struct PortalIn : zeno::INode {
    virtual void complete() override {
        auto name = get_param<std::string>("name");
        std::shared_ptr<Graph> spGraph = getThisGraph();
        assert(spGraph);
        spGraph->portalIns[name] = this->m_name;
    }

    virtual void apply() override {
        auto name = get_param<std::string>("name");
        auto obj = get_input("port");
        std::shared_ptr<Graph> spGraph = getThisGraph();
        assert(spGraph);
        spGraph->portals[name] = std::move(obj);
    }
};

ZENDEFNODE(PortalIn, {
    {{"", "port", "", zeno::Socket_ReadOnly}},
    {},
    {{"string", "name", "RenameMe!"}},
    {"layout"},
});

struct PortalOut : zeno::INode {
    virtual void apply() override {
        auto name = get_param<std::string>("name");
        std::shared_ptr<Graph> spGraph = getThisGraph();
        assert(spGraph);
        auto depnode = zeno::safe_at(spGraph->portalIns, name, "PortalIn");
        spGraph->applyNode(depnode);
        auto obj = zeno::safe_at(spGraph->portals, name, "portal object");
        set_output("port", std::move(obj));
    }
};

ZENDEFNODE(PortalOut, {
    {},
    {"port"},
    {{"string", "name", "RenameMe!"}},
    {"layout"},
});


struct Route : zeno::INode {
    virtual void apply() override {
        if (has_input("input")) {
            auto obj = get_input("input");
            set_output("output", std::move(obj));
        } else {
            set_output("output", std::make_shared<zeno::DummyObject>());
        }
    }
};

ZENDEFNODE(Route, {
    {{"", "input", "", zeno::Socket_ReadOnly}},
    {"output"},
    {},
    {"layout"},
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
    {{"", "object", "", zeno::Socket_ReadOnly}},
    {"newObject"},
    {},
    {"lifecycle"},
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
    {"lifecycle"},
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
    {{"", "object", "", zeno::Socket_ReadOnly}},
    {"newObject"},
    {},
    {"lifecycle"},
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
    {{"", "object", "", zeno::Socket_ReadOnly}},
    {},
    {},
    {"lifecycle"},
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
    {"lifecycle"},
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
    {
        {"", "object", "", zeno::Socket_ReadOnly},
        {"", "data", "", zeno::Socket_ReadOnly},
    },
    {"object"},
    {{"string", "key", ""}},
    {"deprecated"},
});

struct SetUserData2 : zeno::INode {
    virtual void apply() override {
        auto object = get_input("object");
        auto key = get_input2<std::string>("key");
        object->userData().set(key, get_input("data"));
        set_output("object", std::move(object));
    }
};

ZENDEFNODE(SetUserData2, {
    {"object", {"string", "key", ""}, {"string","data",""}},
    {"object"},
    {},
    {"lifecycle"},
});

struct GetUserData : zeno::INode {
    virtual void apply() override {
        auto object = get_input("object");
        auto key = get_param<std::string>("key");
        auto hasValue = object->userData().has(key);
        auto data = hasValue ? object->userData().get(key) : std::make_shared<DummyObject>();
        set_output2("hasValue", hasValue);
        set_output("data", std::move(data));
    }
};

ZENDEFNODE(GetUserData, {
    {{"", "object", "", zeno::Socket_ReadOnly}},
    {"data", {"bool", "hasValue"}},
    {{"string", "key", ""}},
    {"deprecated"},
});

struct GetUserData2 : zeno::INode {
  virtual void apply() override {
    auto object = get_input("object");
    auto key = get_input2<std::string>("key");
    auto hasValue = object->userData().has(key);
    auto data = hasValue ? object->userData().get(key) : std::make_shared<DummyObject>();
    set_output2("hasValue", hasValue);
    set_output("data", std::move(data));
  }
};

ZENDEFNODE(GetUserData2, {
                            {{"", "object", "", zeno::Socket_ReadOnly},
                             {"string", "key", ""}},
                            {"data", {"bool", "hasValue"}},
                            {},
                            {"lifecycle"},
                        });


struct DelUserData : zeno::INode {
    virtual void apply() override {
        auto object = get_input("object");
        auto key = get_param<std::string>("key");
        object->userData().del(key);
    }
};

ZENDEFNODE(DelUserData, {
    {{"", "object", "", zeno::Socket_ReadOnly}},
    {},
    {{"string", "key", ""}},
    {"deprecated"},
});

struct DelUserData2 : zeno::INode {
    virtual void apply() override {
        auto object = get_input("object");
        auto key = get_input2<std::string>("key");
        object->userData().del(key);
        set_output("object", std::move(object));
    }
};

ZENDEFNODE(DelUserData2, {
    {{"string", "key", ""}, {"", "object", "", zeno::Socket_ReadOnly}},
    {"object"},
    {},
    {"lifecycle"},
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
    {
        {"", "dst", "", zeno::Socket_ReadOnly},
        {"", "src", "", zeno::Socket_ReadOnly},
    },
    {"dst"},
    {},
    {"lifecycle"},
});


}
