#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/logger.h>
#include <cstdio>
#include <thread>

namespace {

struct GCTest : zeno::IObjectClone<GCTest, zeno::NumericObject> {
    GCTest() {
        printf("%d GCTest()\n", this->get<int>());
    }

    GCTest(GCTest const &) {
        printf("%d GCTest(GCTest const &)\n", this->get<int>());
    }

    GCTest &operator=(GCTest const &) {
        printf("%d GCTest &operator=(GCTest const &)\n", this->get<int>());
        return *this;
    }

    GCTest(GCTest &&) {
        printf("%d GCTest(GCTest &&)\n", this->get<int>());
    }

    GCTest &operator=(GCTest &&) {
        printf("%d GCTest &operator=(GCTest &&)\n", this->get<int>());
        return *this;
    }

    ~GCTest() {
        printf("%d ~GCTest()\n", this->get<int>());
    }
};


struct MakeGCTest : zeno::INode {
    virtual void apply() override {
        auto obj = std::make_unique<GCTest>();
        obj->set<int>(get_param<int>("value"));
        set_output("value", std::move(obj));
    }
};

ZENDEFNODE(MakeGCTest, {
    {},
    {{"int","value"}},
    {{"int", "value", "42"}},
    {"debug"},
});


struct PrintMessage : zeno::INode {
    virtual void apply() override {
        auto message = get_param<std::string>("message");
        printf("%s\n", message.c_str());
    }
};

ZENDEFNODE(PrintMessage, {
    {},
    {},
    {{"string", "message", "hello-stdout"}},
    {"debug"},
});


struct PrintMessageStdErr : zeno::INode {
    virtual void apply() override {
        auto message = get_param<std::string>("message");
        fprintf(stderr, "%s\n", message.c_str());
    }
};

ZENDEFNODE(PrintMessageStdErr, {
    {},
    {},
    {{"string", "message", "hello-stderr"}},
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
    {{"int", "input", "2"}},
    {{"int", "output"}},
    {},
    {"debug"}
});


struct TriggerExitProcess : zeno::INode {
    virtual void apply() override {
        int status = get_param<int>("status");
        exit(status);
    }
};

ZENDEFNODE(TriggerExitProcess, {
    {},
    {},
    {{"int", "status", "-1"}},
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
        zeno::log_info("{}", get_param<std::string>("message"));
    }
};

ZENDEFNODE(SpdlogInfoMessage, {
    {},
    {},
    {{"string", "message", "hello from spdlog!"}},
    {"debug"},
});


struct SpdlogErrorMessage : zeno::INode {
    virtual void apply() override {
        zeno::log_error("{}", get_param<std::string>("message"));
    }
};

ZENDEFNODE(SpdlogErrorMessage, {
    {},
    {},
    {{"string", "message", "error from spdlog!"}},
    {"debug"},
});


struct TriggerException : zeno::INode {
    virtual void apply() override {
        throw zeno::Exception(get_param<std::string>("message"));
    }
};

ZENDEFNODE(TriggerException, {
    {},
    {},
    {{"string", "message", "exception occurred!"}},
    {"debug"},
});

struct TriggerViewportFault : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        prim->tris.resize(1);
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(TriggerViewportFault, {
    {},
    {"prim"},
    {},
    {"debug"},
});


struct MockRunning : zeno::INode {
    virtual void apply() override {
        int secs = get_input<zeno::NumericObject>("wait seconds")->get<int>();
        std::this_thread::sleep_for(std::chrono::seconds(secs));
    }
};

ZENDEFNODE(MockRunning, {
    {{"list", "SRC"},
     {"int", "wait seconds", "1", zeno::NoSocket, zeno::SpinBox}
    },
    {"DST"},
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
    {{"string", "title", "title"},{"string", "items"},{"vec3f", "background", "0, 0.39, 0.66"},{"vec2f", "size", "500,500"}},
    {},
    {},
    {"layout"},
    });

#if 0
struct CustomNode : zeno::INode {
    virtual void apply() override {

    }
};

ZENDEFINE(CustomNode, {
    {
        {"obj_intput1", zeno::Param_Null, zeno::Socket_ReadOnly},
    },
    {
        {
            {
                "Default",
                {
                    {
                        "Group1",
                        {
                            {"param1", zeno::Param_Null, zeno::Socket_ReadOnly},
                            {"param2", zeno::Param_Prim, zeno::Socket_ReadOnly},
                            {"param3", zeno::Param_Int,  zeno::NoSocket, 2, zeno::Lineedit, {}}
                        }
                    },
                    {
                        "Group2",
                        {
                            {"param4", zeno::Param_String, zeno::Socket_ReadOnly, "", zeno::Multiline, {}},
                            {"param5", zeno::Param_Prim, zeno::Socket_ReadOnly},
                            {"param6", zeno::Param_Null, zeno::NoSocket}
                        }
                    }
                }
            },
            {
                "Default2",
                {
                    {
                        "Group3",
                        {
                            {"param7", zeno::Param_Null, zeno::Socket_ReadOnly},
                            {"param8", zeno::Param_Prim, zeno::Socket_ReadOnly},
                            {"param9", zeno::Param_Null, zeno::NoSocket}
                        }
                    },
                    {
                        "Group4",
                        {
                            {"param10", zeno::Param_Null, zeno::Socket_ReadOnly},
                            {"param11", zeno::Param_Prim, zeno::Socket_ReadOnly},
                            {"param12", zeno::Param_Null, zeno::NoSocket}
                        }
                    }
                }
            },
        }
    },
    {
        {"prim_output1", zeno::Param_Null, zeno::Socket_ReadOnly},
    },
    {
        {"obj_output1", zeno::Param_Null, zeno::Socket_ReadOnly},
    },
    "debug",
    "CUI",
});
#endif

}
