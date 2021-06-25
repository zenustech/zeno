#include <zen/zen.h>
#include <zen/NumericObject.h>
#include <iostream>


struct GCTest : zen::NumericObject {
    GCTest() {
        printf("GCTest()\n");
    }

    ~GCTest() {
        printf("~GCTest()\n");
    }
};


struct MakeGCTest : zen::INode {
    virtual void apply() override {
        auto obj = std::make_unique<GCTest>();
        obj->set<int>(get_param<int>("value"));
        set_output("value", std::move(obj));
    }
};

ZENDEFNODE(MakeGCTest, {
    {},
    {"value"},
    {{"int", "value", "0"}},
    {"debug"},
});
