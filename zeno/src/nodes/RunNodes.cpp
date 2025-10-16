#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/extra/GlobalState.h>

namespace zeno {
namespace {

/*struct RunOnce : zeno::INode {  // deprecated
    virtual void apply() override {
        bool yes = getGlobalState()->substepid == 0;
        auto obj = std::make_unique<zeno::ConditionObject>();
        obj->set(yes);
        ZImpl(set_output("cond", std::move(obj)));
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
        auto obj = std::make_unique<zeno::ConditionObject>();
        obj->set(yes);
        ZImpl(set_output("cond", std::move(obj)));
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
        auto obj = std::make_unique<zeno::ConditionObject>();
        obj->set(yes);
        ZImpl(set_output("cond", std::move(obj)));
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
        auto time = ZImpl(get_input<zeno::NumericObject>("time"))->get<float>();
        ZImpl(getGlobalState())->frame_time = time;
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
        auto time = std::make_unique<zeno::NumericObject>();
        time->set(ZImpl(getGlobalState())->frame_time);
        ZImpl(set_output("time", std::move(time)));
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
        auto time = std::make_unique<zeno::NumericObject>();
        time->set(ZImpl(getGlobalState())->frame_time_elapsed);
        ZImpl(set_output("time", std::move(time)));
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
        auto num = std::make_unique<zeno::NumericObject>();
        num->set(ZImpl(getGlobalState())->getFrameId());
        ZImpl(set_output("FrameNum", std::move(num)));
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
        auto time = std::make_unique<zeno::NumericObject>();
        time->set(ZImpl(getGlobalState())->getFrameId() * ZImpl(getGlobalState())->frame_time
            + ZImpl(getGlobalState())->frame_time_elapsed);
        ZImpl(set_output("time", std::move(time)));
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
        auto portion = std::make_unique<zeno::NumericObject>();
        portion->set(ZImpl(getGlobalState())->frame_time_elapsed / ZImpl(getGlobalState())->frame_time);
        ZImpl(set_output("FramePortion", std::move(portion)));
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
        float dt = ZImpl(getGlobalState())->frame_time;
        if (ZImpl(has_input("desired_dt"))) {
            dt = ZImpl(get_input<zeno::NumericObject>("desired_dt"))->get<float>();
            auto min_scale = ZImpl(get_param<float>("min_scale"));
            dt = std::max(std::fabs(dt), min_scale * ZImpl(getGlobalState())->frame_time);
        }
        if (ZImpl(getGlobalState())->frame_time_elapsed + dt >= ZImpl(getGlobalState())->frame_time) {
            dt = ZImpl(getGlobalState())->frame_time - ZImpl(getGlobalState())->frame_time_elapsed;
            ZImpl(getGlobalState())->frame_time_elapsed = ZImpl(getGlobalState())->frame_time;
            ZImpl(getGlobalState())->has_frame_completed = true;
        } else {
            ZImpl(getGlobalState())->frame_time_elapsed += dt;
        }
        ZImpl(getGlobalState())->time_step_integrated = true;
        auto ret = std::make_unique<zeno::NumericObject>();
        ret->set(dt);
        ZImpl(set_output("actual_dt", std::move(ret)));
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
