#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/extra/GlobalState.h>

namespace zeno {
namespace {

/*struct RunOnce : zeno::INode {  // deprecated
    virtual void apply() override {
        bool yes = getGlobalState()->substepid == 0;
        auto obj = std::make_shared<zeno::ConditionObject>();
        obj->set(yes);
        set_output("cond", std::move(obj));
    }
};

ZENDEFNODE(RunOnce, {
    {},
    {"cond"},
    {},
    {"frame"},
});

struct RunAfterFrame : zeno::INode {  // deprecated
    virtual void apply() override {
        bool yes = getGlobalState()->has_frame_completed || !getGlobalState()->time_step_integrated;
        auto obj = std::make_shared<zeno::ConditionObject>();
        obj->set(yes);
        set_output("cond", std::move(obj));
    }
};

ZENDEFNODE(RunAfterFrame, {
    {},
    {"cond"},
    {},
    {"frame"},
});

struct RunBeforeFrame : zeno::INode {  // deprecated
    virtual void apply() override {
        bool yes = !getGlobalState()->has_substep_executed;
        auto obj = std::make_shared<zeno::ConditionObject>();
        obj->set(yes);
        set_output("cond", std::move(obj));
    }
};

ZENDEFNODE(RunBeforeFrame, {
    {},
    {"cond"},
    {},
    {"frame"},
});*/


struct SetFrameTime : zeno::INode {
    virtual void apply() override {
        auto time = get_input<zeno::NumericObject>("time")->get<float>();
        getGlobalState()->frame_time = time;
    }
};

ZENDEFNODE(SetFrameTime, {
    {{gParamType_Float, "time", "", zeno::Socket_ReadOnly}},
    {},
    {},
    {"frame"},
});

struct GetFrameTime : zeno::INode {
    virtual void apply() override {
        auto time = std::make_shared<zeno::NumericObject>();
        time->set(getGlobalState()->frame_time);
        set_output("time", std::move(time));
    }
};

ZENDEFNODE(GetFrameTime, {
    {},
    {{gParamType_Float,"time"}},
    {},
    {"frame"},
});

struct GetFrameTimeElapsed : zeno::INode {
    virtual void apply() override {
        auto time = std::make_shared<zeno::NumericObject>();
        time->set(getGlobalState()->frame_time_elapsed);
        set_output("time", std::move(time));
    }
};

ZENDEFNODE(GetFrameTimeElapsed, {
    {},
    {{gParamType_Float,"time"}},
    {},
    {"frame"},
});

struct GetFrameNum : zeno::INode {
    virtual void apply() override {
        auto num = std::make_shared<zeno::NumericObject>();
        num->set(getGlobalState()->getFrameId());
        set_output("FrameNum", std::move(num));
    }
};

ZENDEFNODE(GetFrameNum, {
    {},
    {{gParamType_Int,"FrameNum"}},
    {},
    {"frame"},
});

struct GetTime : zeno::INode {
    virtual void apply() override {
        auto time = std::make_shared<zeno::NumericObject>();
        time->set(getGlobalState()->getFrameId() * getGlobalState()->frame_time
            + getGlobalState()->frame_time_elapsed);
        set_output("time", std::move(time));
    }
};

ZENDEFNODE(GetTime, {
    {},
    {{gParamType_Float,"time"}},
    {},
    {"frame"},
});

struct GetFramePortion : zeno::INode {
    virtual void apply() override {
        auto portion = std::make_shared<zeno::NumericObject>();
        portion->set(getGlobalState()->frame_time_elapsed / getGlobalState()->frame_time);
        set_output("FramePortion", std::move(portion));
    }
};

ZENDEFNODE(GetFramePortion, {
    {},
    {{gParamType_Float,"FramePortion"}},
    {},
    {"frame"},
});

struct IntegrateFrameTime : zeno::INode {
    virtual void apply() override {
        float dt = getGlobalState()->frame_time;
        if (has_input("desired_dt")) {
            dt = get_input<zeno::NumericObject>("desired_dt")->get<float>();
            auto min_scale = get_param<float>("min_scale");
            dt = std::max(std::fabs(dt), min_scale * getGlobalState()->frame_time);
        }
        if (getGlobalState()->frame_time_elapsed + dt >= getGlobalState()->frame_time) {
            dt = getGlobalState()->frame_time - getGlobalState()->frame_time_elapsed;
            getGlobalState()->frame_time_elapsed = getGlobalState()->frame_time;
            getGlobalState()->has_frame_completed = true;
        } else {
            getGlobalState()->frame_time_elapsed += dt;
        }
        getGlobalState()->time_step_integrated = true;
        auto ret = std::make_shared<zeno::NumericObject>();
        ret->set(dt);
        set_output("actual_dt", std::move(ret));
    }
};

ZENDEFNODE(IntegrateFrameTime, {
    {{gParamType_Float, "desired_dt", "", zeno::Socket_ReadOnly}},
    {{gParamType_Float,"actual_dt"}},
    {{gParamType_Float, "min_scale", "0.0001"}},
    {"frame"},
});

}
}
