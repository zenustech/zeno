#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/logger.h>
#include <cstdio>

namespace {

struct GCTest : zeno::NumericObject {
    GCTest() {
        printf("GCTest()\n");
    }

    ~GCTest() {
        printf("~GCTest()\n");
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
    {{"string", "message", "hello"}},
    {"debug"},
});


struct ExitProcess : zeno::INode {
    virtual void apply() override {
        int status = get_param<int>("status");
        exit(status);
    }
};

ZENDEFNODE(ExitProcess, {
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
        x /= x;
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


}
