#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/IGeometryObject.h>
#include <zeno/utils/logger.h>
#include <cstdio>
#include <thread>


namespace zeno {

struct PrintMessage : zeno::INode {
    virtual void apply() override {
        auto message = ZImpl(get_param<std::string>("message"));
        printf("%s\n", message.c_str());
    }
};

ZENDEFNODE(PrintMessage, {
    {},
    {},
    {{gParamType_String, "message", "hello-stdout"}},
    {"debug"},
});


struct PrintMessageStdErr : zeno::INode {
    virtual void apply() override {
        auto message = ZImpl(get_param<std::string>("message"));
        fprintf(stderr, "%s\n", message.c_str());
    }
};

ZENDEFNODE(PrintMessageStdErr, {
    {},
    {},
    {{gParamType_String, "message", "hello-stderr"}},
    {"debug"},
});


struct InfiniteLoop : zeno::INode {
    virtual void apply() override {
        while (true) {
            int j;
            j = 0;
        }
    }
};

ZENDEFNODE(InfiniteLoop, {
    {{gParamType_Int, "input", "2"}},
    {{gParamType_Int, "output"}},
    {},
    {"debug"}
});


struct TriggerExitProcess : zeno::INode {
    virtual void apply() override {
        int status = ZImpl(get_param<int>("status"));
        exit(status);
    }
};

ZENDEFNODE(TriggerExitProcess, {
    {},
    {},
    {{gParamType_Int, "status", "-1"}},
    {"debug"},
});


struct TriggerSegFault : zeno::INode {
    virtual void apply() override {
        *(volatile float *)nullptr = 0;
    }
};

ZENDEFNODE(TriggerSegFault, {
    {},
    {},
    {},
    {"debug"},
});


struct TriggerDivideZero : zeno::INode {
    virtual void apply() override {
        volatile int x = 0;
        x = x / x;
    }
};

ZENDEFNODE(TriggerDivideZero, {
    {},
    {},
    {},
    {"debug"},
});


struct TriggerAbortSignal : zeno::INode {
    virtual void apply() override {
        abort();
    }
};

ZENDEFNODE(TriggerAbortSignal, {
    {},
    {},
    {},
    {"debug"},
});



struct SpdlogInfoMessage : zeno::INode {
    virtual void apply() override {
        zeno::log_info("{}", ZImpl(get_param<std::string>("message")));
    }
};

ZENDEFNODE(SpdlogInfoMessage, {
    {},
    {},
    {{gParamType_String, "message", "hello from spdlog!"}},
    {"debug"},
});


struct SpdlogErrorMessage : zeno::INode {
    virtual void apply() override {
        zeno::log_error("{}", ZImpl(get_param<std::string>("message")));
    }
};

ZENDEFNODE(SpdlogErrorMessage, {
    {},
    {},
    {{gParamType_String, "message", "error from spdlog!"}},
    {"debug"},
});


struct TriggerException : zeno::INode {
    virtual void apply() override {
        throw zeno::Exception(ZImpl(get_param<std::string>("message")));
    }
};

ZENDEFNODE(TriggerException, {
    {},
    {},
    {{gParamType_String, "message", "exception occurred!"}},
    {"debug"},
});

struct TriggerViewportFault : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_unique<zeno::PrimitiveObject>();
        prim->tris.resize(1);
        ZImpl(set_output("prim", std::move(prim)));
    }
};

ZENDEFNODE(TriggerViewportFault, {
    {},
    {{gParamType_Primitive, "prim"}},
    {},
    {"debug"},
});


struct MockRunning : zeno::INode {
    virtual void apply() override {
        int secs = ZImpl(get_input<zeno::NumericObject>("wait seconds"))->get<int>();
        if (get_input2_bool("cause exception")) {
            throw makeError<UnimplError>(": MockRunning");
        }
        std::this_thread::sleep_for(std::chrono::seconds(secs));
        auto geom = zeno::create_GeometryObject(zeno::Topo_IndiceMesh, true, { zeno::vec3f(0,0,0) }, {});
        set_output("DST", std::move(geom));
    }
};

ZENDEFNODE(MockRunning, {
    {{gParamType_List, "SRC"},
     {gParamType_Int, "wait seconds", "1", zeno::NoSocket, zeno::Lineedit},
     {gParamType_Bool, "cause exception", "0"}
    },
    {{gParamType_IObject, "DST"}},
    {},
    {"debug"},
});


struct Blackboard : zeno::INode {
    virtual void apply() override {
    }
};

ZENDEFNODE(Blackboard, {
    {},
    {},
    {},
    {"layout"},
});

struct Group : zeno::INode {
    virtual void apply() override {
    }
};

ZENDEFNODE(Group, {
    {{gParamType_String, "title", "title"},{gParamType_String, "items"},{gParamType_Vec3f, "background", "0, 0.39, 0.66"},{gParamType_Vec2f, "size", "500,500"}},
    {},
    {},
    {"layout"},
    });

struct CustomNode : zeno::INode {
    virtual void apply() override {

    }
};

ZENO_CUSTOMUI_NODE(CustomNode,
    zeno::ObjectParams{
         zeno::ParamObject("Input", gParamType_Geometry)
    },
    zeno::CustomUIParams{
        zeno::ParamTab {
            "Tab1",
            {
                zeno::ParamGroup {
                    "Group1",
                    zeno::PrimitiveParams {
                        zeno::ParamPrimitive("Int Val", gParamType_Int, 2),
                        zeno::ParamPrimitive("String Val", gParamType_String, "abc")
                    }
                },
                zeno::ParamGroup {
                    "Group2",
                    zeno::PrimitiveParams {
                        zeno::ParamPrimitive("Float Val", gParamType_Float, 3.2f),
                        zeno::ParamPrimitive("Items", gParamType_String, "Item 1", zeno::Combobox, std::vector<std::string>{"Item1", "Item2", "Item3"})
                    }
                }
            }
        }
    },
    /*output prims:*/
    zeno::PrimitiveParams{
        zeno::ParamPrimitive("output prim", gParamType_Int, 3),
    },
    /*output objects:*/
    zeno::ObjectParams{
        zeno::ParamObject("obj_output", gParamType_Geometry)
    },
    zeno::NodeUIStyle{ "", "" },
    "debug",     //category
    "",     //nickname
    ""      //doc
);

}
