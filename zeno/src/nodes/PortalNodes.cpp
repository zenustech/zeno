#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/types/DummyObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/safe_at.h>
#include <zeno/core/Graph.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/DictObject.h>

namespace zeno {

#if 0
struct PortalIn : zeno::INode {
    virtual void complete() override {
        auto name = ZImpl(get_param<std::string>("name"));
        std::shared_ptr<Graph> spGraph = getThisGraph();
        assert(spGraph);
        spGraph->portalIns[name] = this->get_name();
    }

    virtual void apply() override {
        auto name = ZImpl(get_param<std::string>("name"));
        auto obj = ZImpl(clone_input("port"));
        std::shared_ptr<Graph> spGraph = getThisGraph();
        assert(spGraph);
        spGraph->portals[name] = std::move(obj);
    }
};

ZENDEFNODE(PortalIn, {
    {{gParamType_IObject, "port", "", zeno::Socket_ReadOnly}},
    {},
    {{gParamType_String, "name", "RenameMe!"}},
    {"layout"},
});

struct PortalOut : zeno::INode {
    virtual void apply() override {
        auto name = ZImpl(get_param<std::string>("name"));
        std::shared_ptr<Graph> spGraph = getThisGraph();
        assert(spGraph);
        auto depnode = zeno::safe_at(spGraph->portalIns, name, "PortalIn");
        spGraph->applyNode(depnode);
        auto obj = zeno::safe_at(spGraph->portals, name, "portal object");
        ZImpl(set_output("port", std::move(obj)));
    }
};

ZENDEFNODE(PortalOut, {
    {},
    {{gParamType_IObject, "port"}},
    {{gParamType_String, "name", "RenameMe!"}},
    {"layout"},
});
#endif


struct Route : zeno::INode {
    virtual void apply() override {
        if (ZImpl(has_input("input"))) {
            auto obj = ZImpl(clone_input("input"));
            ZImpl(set_output("output", std::move(obj)));
        } else {
            ZImpl(set_output("output", std::make_unique<zeno::DummyObject>()));
        }
    }
    CustomUI export_customui() const override {
        CustomUI ui = INode::export_customui();
        ui.uistyle.background = "#99B2C4";
        return ui;
    }
};

ZENDEFNODE(Route, {
    {
        {gParamType_IObject, "input", "", zeno::Socket_ReadOnly},
        {gParamType_Float, "float_val", "0"},
        {gParamType_Int, "int_val", "0"},
        {gParamType_String, "string_val", "0"},
        {gParamType_Vec3f, "vec3f_val", "0,0,0"}
    },
    {{gParamType_IObject, "output"}},
    {{"enum normal lightCamera material matrix", "RunType", "normal"}},
    {"layout"},
});


struct Stamp : zeno::INode {
    virtual void apply() override {
        if (ZImpl(has_input("input"))) {
            auto obj = ZImpl(clone_input("input"));
            ZImpl(set_output("output", std::move(obj)));
        }
        else {
            ZImpl(set_output("output", std::make_unique<zeno::DummyObject>()));
        }
    }
};

ZENDEFNODE(Stamp, {
    {{gParamType_IObject, "input"}},
    {{gParamType_IObject, "output"}},
    {{"enum UnChanged DataChange ShapeChange TotalChange", "mode", "UnChanged"},
     {gParamType_String, "name", ""}},
    {"lifecycle"}
});


struct Clone : zeno::INode {
    virtual void apply() override {
        auto obj = ZImpl(clone_input("object"));
        auto newobj = obj->clone();
        if (!newobj) {
            log_error("requested object doesn't support clone");
            return;
        }
        ZImpl(set_output("newObject", std::move(newobj)));
        ZImpl(set_output("origin", std::move(obj)));
    }
};

ZENDEFNODE(Clone, {
    {{gParamType_IObject, "object", "", zeno::Socket_ReadOnly}},
    {
        {"object", "newObject"},
        {"object", "origin"},
    },
    {},
    {"lifecycle"},
});


struct Assign : zeno::INode {
    virtual void apply() override {
        auto src = ZImpl(clone_input("src"));
        auto dst = ZImpl(clone_input("dst"));
        *dst = *src;
        ZImpl(set_output("dst", std::move(dst)));
    }
};

ZENDEFNODE(Assign, {
    {{"object", "dst"}, {"object", "src"} },
    {{"object", "dst"}},
    {},
    {"lifecycle"},
});

struct SetUserData : zeno::INode {
    virtual void apply() override {
        auto object = ZImpl(clone_input("object"));
        auto key = zsString2Std(get_input2_string("key"));
        UserData* pUsrData = dynamic_cast<UserData*>(object->userData());
        pUsrData->set(key, ZImpl(clone_input("data")));
        ZImpl(set_output("object", std::move(object)));
    }
};

ZENDEFNODE(SetUserData, {
    {
        {gParamType_IObject, "object", "", zeno::Socket_ReadOnly},
        {gParamType_IObject, "data", "", zeno::Socket_ReadOnly},
    },
    {{gParamType_IObject, "object"}},
    {{gParamType_String, "key", ""}},
    {"deprecated"},
});

struct SetUserData2 : zeno::INode {
    virtual void apply() override {
        auto object = ZImpl(clone_input("object"));
        auto key = zsString2Std(get_input2_string("key"));
        UserData* pUsrData = dynamic_cast<UserData*>(object->userData());
        pUsrData->set(key, ZImpl(clone_input("data")));
        ZImpl(set_output("object", std::move(object)));
    }
};

ZENDEFNODE(SetUserData2, {
    {
        {gParamType_IObject, "object"},
        {gParamType_String, "key", ""},
        {gParamType_AnyNumeric, "data", "0"}
    },
    {
        {gParamType_IObject, "object"}
    },
    {},
    {"lifecycle"},
});

struct GetUserData : zeno::INode {
    virtual void apply() override {
        auto object = ZImpl(clone_input("object"));
        auto key = zsString2Std(get_input2_string("key"));
        UserData* pUsrData = dynamic_cast<UserData*>(object->userData());
        auto hasValue = pUsrData->has(key);
        if (hasValue) {
            ZImpl(set_primitive_output("hasValue", hasValue));
            ZImpl(set_output("data", pUsrData->get(key)));
        }
        else {
            ZImpl(set_primitive_output("hasValue", false));
            ZImpl(set_output("data", std::make_unique<DummyObject>()));
        }
    }
};

ZENDEFNODE(GetUserData, {
    {{gParamType_IObject, "object", "", zeno::Socket_ReadOnly}},
    {{gParamType_IObject, "data"}, {gParamType_Bool, "hasValue"}},
    {{gParamType_String, "key", ""}},
    {"deprecated"},
});

struct GetUserData2 : zeno::INode {
  virtual void apply() override {
    auto object = ZImpl(clone_input("object"));
    auto key = zsString2Std(get_input2_string("key"));
    UserData* pUsrData = dynamic_cast<UserData*>(object->userData());
    auto hasValue = pUsrData->has(key);
    auto data = hasValue ? pUsrData->get(key) : std::make_unique<DummyObject>();
    ZImpl(set_primitive_output("hasValue", hasValue));
    ZImpl(set_output("data", std::move(data)));
  }
};

ZENDEFNODE(GetUserData2, {
                            {{gParamType_IObject, "object", "", zeno::Socket_ReadOnly},
                             {gParamType_String, "key", ""}},
                            {{gParamType_IObject, "data"}, {gParamType_Bool, "hasValue"}},
                            {},
                            {"lifecycle"},
                        });

struct GetUserData3 : zeno::INode {
    virtual void apply() override {
        auto object = ZImpl(clone_input("object"));
        auto key = zsString2Std(get_input2_string("key"));
        UserData* pUsrData = static_cast<UserData*>(object->userData());
        if (pUsrData->has(key)) {
            auto data = pUsrData->get(key);
            if (auto numericObj = dynamic_cast<NumericObject*>(data.get())) {
                std::visit([&](auto&& val) {
                    using T = std::decay_t<decltype(val)>;
                    if (std::is_same_v<T, int>) {
                        ZImpl(set_primitive_output("data", std::get<int>(numericObj->get())));
                    }
                    else if (std::is_same_v<T, float>) {
                        ZImpl(set_primitive_output("data", std::get<float>(numericObj->get())));
                    }
                    else if (std::is_same_v<T, vec2i>) {
                        ZImpl(set_primitive_output("data", std::get<vec2i>(numericObj->get())));
                    }
                    else if (std::is_same_v<T, vec3i>) {
                        ZImpl(set_primitive_output("data", std::get<vec3i>(numericObj->get())));
                    }
                    else if (std::is_same_v<T, vec4i>) {
                        ZImpl(set_primitive_output("data", std::get<vec4i>(numericObj->get())));
                    }
                    else if (std::is_same_v<T, vec2f>) {
                        ZImpl(set_primitive_output("data", std::get<vec2f>(numericObj->get())));
                    }
                    else if (std::is_same_v<T, vec3f>) {
                        ZImpl(set_primitive_output("data", std::get<vec3f>(numericObj->get())));
                    }
                    else if (std::is_same_v<T, vec4f>) {
                        ZImpl(set_primitive_output("data", std::get<vec4f>(numericObj->get())));
                    }
                }, numericObj->get());
            }
            else if (auto stringObj = dynamic_cast<StringObject*>(object.get())) {
                ZImpl(set_primitive_output("data", stringObj->get()));
            }
            else {
                throw makeError<UnimplError>("ObjectToBasetype expect not numericObject or stringObject");
            }
        }
        else {
            throw makeError<UnimplError>("only support numeric or string");
        }
    }
};

ZENDEFNODE(GetUserData3, {
    {
        {gParamType_IObject, "object"},
        {gParamType_String, "key", ""},
    },
    {
        {gParamType_AnyNumeric, "data"},
    },
    {},
    {"lifecycle"},
});


struct DelUserData : zeno::INode {
    virtual void apply() override {
        auto object = ZImpl(clone_input("object"));
        auto key = zsString2Std(get_input2_string("key"));
        UserData* pUsrData = dynamic_cast<UserData*>(object->userData());
        pUsrData->del(stdString2zs(key));
    }
};

ZENDEFNODE(DelUserData, {
    {{gParamType_IObject, "object", "", zeno::Socket_ReadOnly}},
    {},
    {{gParamType_String, "key", ""}},
    {"deprecated"},
});

struct DelUserData2 : zeno::INode {
    virtual void apply() override {
        auto object = ZImpl(clone_input("object"));
        auto key = zsString2Std(get_input2_string("key"));
        UserData* pUsrData = dynamic_cast<UserData*>(object->userData());
        pUsrData->del(stdString2zs(key));
        ZImpl(set_output("object", std::move(object)));
    }
};

ZENDEFNODE(DelUserData2, {
    {{gParamType_String, "key", ""}, {gParamType_IObject, "object", "", zeno::Socket_ReadOnly}},
    {{gParamType_IObject, "object"}},
    {},
    {"lifecycle"},
});

struct ObjectToPrimInt : zeno::INode {
    virtual void apply() override {
        auto object = ZImpl(clone_input("object"));
        if (auto numericObj = dynamic_cast<NumericObject*>(object.get())) {
            int val = 0;
            std::visit([&val](auto&& var) {
                using T = std::decay_t<decltype(var)>;
                if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float>) {
                    val = var;
                }
            }, numericObj->get());
            ZImpl(set_primitive_output("value", val));
        }
        else {
            ZImpl(set_primitive_output("value", -1));
        }
    }
};

ZENDEFNODE(ObjectToPrimInt, {
    {{gParamType_IObject, "object", "", zeno::Socket_ReadOnly}},
    {{gParamType_Int, "value", "0"}},
    {},
    {"objToNumeric"},
    });

struct ObjectToPrimString : zeno::INode {
    virtual void apply() override {
        auto object = ZImpl(clone_input("object"));
        if (auto stringObj = dynamic_cast<StringObject*>(object.get())) {
            ZImpl(set_primitive_output("value", stringObj->get()));
        }
        else {
            ZImpl(set_primitive_output("value", ""));
        }
    }
};

ZENDEFNODE(ObjectToPrimString, {
    {{gParamType_IObject, "object", "", zeno::Socket_ReadOnly}},
    {{gParamType_String, "value", ""}},
    {},
    {"objToString"},
    });

struct ObjectToBasetype : zeno::INode {
    virtual void apply() override {
        auto object = ZImpl(clone_input("Object"));
        auto type = zsString2Std(get_input2_string("Output type"));
        if (auto numericObj = dynamic_cast<NumericObject*>(object.get())) {
            std::visit([&type](auto&& val) {
                using T = std::decay_t<decltype(val)>;
                std::string valtype;
                if (std::is_same_v<T, int>) {
                    valtype = "int";
                } else if (std::is_same_v<T, float>) {
                    valtype = "float";
                } else if (std::is_same_v<T, vec2i>) {
                    valtype = "vec2i";
                } else if (std::is_same_v<T, vec3i>) {
                    valtype = "vec3i";
                } else if (std::is_same_v<T, vec4i>) {
                    valtype = "vec4i";
                } else if (std::is_same_v<T, vec2f>) {
                    valtype = "vec2f";
                } else if (std::is_same_v<T, vec3f>) {
                    valtype = "vec3f";
                } else if (std::is_same_v<T, vec4f>) {
                    valtype = "vec4f";
                }
                if (type != valtype) {
                    valtype = "none";
                    throw makeError<UnimplError>("ObjectToBasetype expect " + type + " got " + valtype);
                }
            }, numericObj->get());
            if (type == "int") {
                ZImpl(set_primitive_output("int", std::get<int>(numericObj->get())));
            } else if (type == "float") {
                ZImpl(set_primitive_output("float", std::get<float>(numericObj->get())));
            } else if (type == "vec2i") {
                ZImpl(set_primitive_output("vec2i", std::get<vec2i>(numericObj->get())));
            } else if (type == "vec3i") {
                ZImpl(set_primitive_output("vec3i", std::get<vec3i>(numericObj->get())));
            } else if (type == "vec4i") {
                ZImpl(set_primitive_output("vec4i", std::get<vec4i>(numericObj->get())));
            } else if (type == "vec2f") {
                ZImpl(set_primitive_output("vec2f", std::get<vec2f>(numericObj->get())));
            } else if (type == "vec3f") {
                ZImpl(set_primitive_output("vec3f", std::get<vec3f>(numericObj->get())));
            } else if (type == "vec4f") {
                ZImpl(set_primitive_output("vec4f", std::get<vec4f>(numericObj->get())));
            }
        } else if (auto stringObj = dynamic_cast<StringObject*>(object.get())) {
            if (type == "string") {
                ZImpl(set_primitive_output("string", stringObj->get()));
            } else {
                throw makeError<UnimplError>("ObjectToBasetype expect " + type + " got stringObject");
            }
        } else {
            throw makeError<UnimplError>("ObjectToBasetype expect not numericObject or stringObject");
        }
    }
};

ZENDEFNODE(ObjectToBasetype, {
    {
        {gParamType_IObject, "object", "", zeno::Socket_ReadOnly},
        {"enum int float string vec2i vec3i vec4i vec2f vec3f vec4f", "Output type", "int"},
        //ParamPrimitive("Output type", gParamType_String, "int", zeno::Combobox, std::vector<std::string>{"int", "float", "string", "vec2i", "vec3i", "vec4i", "vec2f", "vec3f", "vec4f"}),
    },
    {
        {gParamType_Int, "int", "0"}, 
        {gParamType_Float, "float", "0"},
        {gParamType_String, "string", ""},
        {gParamType_Vec2f, "vec2i", "0,0"},
        {gParamType_Vec3f, "vec3i", "0,0,0"},
        {gParamType_Vec4f, "vec4i", "0,0,0,0"},
        {gParamType_Vec2f, "vec2f", "0,0"},
        {gParamType_Vec3f, "vec3f", "0,0,0"},
        {gParamType_Vec4f, "vec4f", "0,0,0,0"},
    },
    {},
    {"ObjectToBasetype"},
    });

#if 0
struct CopyAllUserData : zeno::INode {
    virtual void apply() override {
        auto src = ZImpl(clone_input("src"));
        auto dst = ZImpl(clone_input("dst"));
        dst->userData() = src->userData();
        ZImpl(set_output("dst", std::move(dst)));
    }
};

ZENDEFNODE(CopyAllUserData, {
    {
        {gParamType_IObject,  "dst", "", zeno::Socket_ReadOnly},
        {gParamType_IObject,  "src", "", zeno::Socket_ReadOnly},
    },
    {{gParamType_IObject, "dst"}},
    {},
    {"lifecycle"},
});
#endif

}

