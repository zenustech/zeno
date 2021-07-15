#include <zeno/zeno.h>
#include <zeno/NumericObject.h>
#include <cstdio>


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
