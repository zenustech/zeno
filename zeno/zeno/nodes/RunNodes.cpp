#ifdef ZENO_GLOBALSTATE
#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/ConditionObject.h>
#include <zeno/extra/GlobalState.h>

/*struct RunOnce : zeno::INode {  // deprecated
    virtual void apply() override {
        bool yes = zeno::state.substepid == 0;
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
        bool yes = zeno::state.has_frame_completed || !zeno::state.time_step_integrated;
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
        bool yes = !zeno::state.has_substep_executed;
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
        zeno::state.frame_time = time;
    }
};

ZENDEFNODE(SetFrameTime, {
    {"time"},
    {},
    {},
    {"frame"},
});

struct GetFrameTime : zeno::INode {
    virtual void apply() override {
        auto time = std::make_shared<zeno::NumericObject>();
        time->set(zeno::state.frame_time);
        set_output("time", std::move(time));
    }
};

ZENDEFNODE(GetFrameTime, {
    {},
    {"time"},
    {},
    {"frame"},
});

struct GetFrameTimeElapsed : zeno::INode {
    virtual void apply() override {
        auto time = std::make_shared<zeno::NumericObject>();
        time->set(zeno::state.frame_time_elapsed);
        set_output("time", std::move(time));
    }
};

ZENDEFNODE(GetFrameTimeElapsed, {
    {},
    {"time"},
    {},
    {"frame"},
});

struct GetFrameNum : zeno::INode {
    virtual void apply() override {
        auto num = std::make_shared<zeno::NumericObject>();
        num->set(zeno::state.frameid);
        set_output("FrameNum", std::move(num));
    }
};

ZENDEFNODE(GetFrameNum, {
    {},
    {"FrameNum"},
    {},
    {"frame"},
});

struct GetTime : zeno::INode {
    virtual void apply() override {
        auto time = std::make_shared<zeno::NumericObject>();
        time->set(zeno::state.frameid * zeno::state.frame_time
            + zeno::state.frame_time_elapsed);
        set_output("time", std::move(time));
    }
};

ZENDEFNODE(GetTime, {
    {},
    {"time"},
    {},
    {"frame"},
});

struct GetFramePortion : zeno::INode {
    virtual void apply() override {
        auto portion = std::make_shared<zeno::NumericObject>();
        portion->set(zeno::state.frame_time_elapsed / zeno::state.frame_time);
        set_output("FramePortion", std::move(portion));
    }
};

ZENDEFNODE(GetFramePortion, {
    {},
    {"FramePortion"},
    {},
    {"frame"},
});

struct IntegrateFrameTime : zeno::INode {
    virtual void apply() override {
        float dt = zeno::state.frame_time;
        if (has_input("desired_dt")) {
            dt = get_input<zeno::NumericObject>("desired_dt")->get<float>();
            auto min_scale = get_param<float>("min_scale");
            dt = std::max(std::fabs(dt), min_scale * zeno::state.frame_time);
        }
        if (zeno::state.frame_time_elapsed + dt >= zeno::state.frame_time) {
            dt = zeno::state.frame_time - zeno::state.frame_time_elapsed;
            zeno::state.frame_time_elapsed = zeno::state.frame_time;
            zeno::state.has_frame_completed = true;
        } else {
            zeno::state.frame_time_elapsed += dt;
        }
        zeno::state.time_step_integrated = true;
        auto ret = std::make_shared<zeno::NumericObject>();
        ret->set(dt);
        set_output("actual_dt", std::move(ret));
    }
};

ZENDEFNODE(IntegrateFrameTime, {
    {"desired_dt"},
    {"actual_dt"},
    {{"float", "min_scale", "0.0001"}},
    {"frame"},
});
#endif
