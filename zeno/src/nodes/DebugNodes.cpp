#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/logger.h>
#include <cstdio>

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
    {"value"},
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


}
