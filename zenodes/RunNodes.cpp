#include <zen/zen.h>
#include <zen/ConditionObject.h>
#include <zen/GlobalState.h>

struct RunOnce : zen::INode {
    virtual void apply() override {
        bool yes = zen::state.substepid == 0;
        auto obj = std::make_unique<zen::ConditionObject>();
        obj->set(yes);
        set_output("cond", std::move(obj));
    }
};

ZENDEFNODE(RunOnce, {
    {},
    {"cond"},
    {},
    {"substep"},
});
