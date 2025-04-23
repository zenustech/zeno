#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/types/DummyObject.h>
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
    {"layout"},
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
    {"input"},
    {"output"},
    {},
    {"layout"},
});


struct Stamp : zeno::INode {
    virtual void apply() override {
        if (has_input("input")) {
            auto obj = get_input("input");
            auto session = &zeno::getSession();
            if (has_input("stampMode")) {
                int currframe = session->globalState->frameid;
                int beginframe = session->globalComm->beginFrameNumber;
                std::string mode = get_input2<std::string>("stampMode");
                if (mode == "UnChanged") {
                    if (currframe != beginframe) {
                        obj = session->globalComm->constructEmptyObj(inputObjType);
                    }
                    obj->userData().set2("stamp-change", "UnChanged");
                } else if (mode == "TotalChange") {
                    obj->userData().set2("stamp-change", "TotalChange");
                } else if (mode == "DataChange") {
                    obj->userData().set2("stamp-change", "DataChange");
                    std::string changehint = get_input2<std::string>("changeHint");
                    obj->userData().set2("stamp-dataChange-hint", changehint);
                } else if (mode == "ShapeChange") {
                    obj->userData().set2("stamp-change", "TotalChange");//shapechange暂时全部按Totalchange处理
                }
            }
            inputObjType = session->globalComm->getObjType(obj);
            auto &ud = obj->userData();
            if (!ud.has<std::string>("ResourceType")) {
                ud.set2("ResourceType", get_input2<std::string>("ResourceType"));
            }
            if (!ud.has<std::string>("ObjectName")) {
                ud.set2("ObjectName", get_input2<std::string>("ObjectName"));
            }
            set_output("output", std::move(obj));
        }
        else {
            set_output("output", std::make_shared<zeno::DummyObject>());
        }
    }

    int inputObjType = 0;
};

ZENDEFNODE(Stamp, {
    {
        "input",
        {"enum UnChanged DataChange ShapeChange TotalChange", "stampMode", "UnChanged"},
        {"enum Mesh Matrixes SceneDescriptor", "ResourceType", "Mesh"},
        {"string", "ObjectName", ""},
        {"string", "changeHint", ""}
    },
    {"output"},
    {
        //{"string", "name", ""}
    },
    {"lifecycle"}
});

struct SetToMatrix : zeno::INode {
    virtual void apply() override {
        if (has_input("input")) {
            auto obj = get_input("input");
            set_output("output", std::move(obj));
        }
        else {
            set_output("output", std::make_shared<zeno::DummyObject>());
        }
    }
};

ZENDEFNODE(SetToMatrix, {
    {"input"},
    {"output"},
    {},
    {"lifecycle"},
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
        set_output("origin", obj);
    }
};

ZENDEFNODE(Clone, {
    {"object"},
    {
        "newObject",
        "origin",
    },
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
    {"object"},
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
    {"object"},
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
    {"object", "data"},
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
    {"object"},
    {"data", {"bool", "hasValue"}},
    {{"string", "key", ""}},
    {"deprecated"},
});

struct GetUserData2 : zeno::INode {
  virtual void apply() override {
    std::shared_ptr<IObject> default_value = std::make_shared<DummyObject>();
    if (has_input("default_value")) {
        default_value = get_input("default_value");
    }
    auto object = get_input("object");
    auto key = get_input2<std::string>("key");
    auto hasValue = object->userData().has(key);
    auto data = hasValue ? object->userData().get(key) : default_value->clone();
    set_output2("hasValue", hasValue);
    set_output("data", std::move(data));
  }
};

ZENDEFNODE(GetUserData2, {
                            {
                                "object",
                                {"string", "key", ""},
                                "default_value",
                            },
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
    {"object"},
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
    {{"string", "key", ""}, "object"},
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
    {"dst", "src"},
    {"dst"},
    {},
    {"lifecycle"},
});


}
