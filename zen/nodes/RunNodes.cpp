#include <zen/zen.h>
#include <zen/NumericObject.h>
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
    {"runtime"},
});

struct RunAfterFrame : zen::INode {
    virtual void apply() override {
        bool yes = zen::state.has_frame_completed || !zen::state.time_step_integrated;
        auto obj = std::make_unique<zen::ConditionObject>();
        obj->set(yes);
        set_output("cond", std::move(obj));
    }
};

ZENDEFNODE(RunAfterFrame, {
    {},
    {"cond"},
    {},
    {"runtime"},
});

struct RunBeforeFrame : zen::INode {
    virtual void apply() override {
        bool yes = !zen::state.has_substep_executed;
        auto obj = std::make_unique<zen::ConditionObject>();
        obj->set(yes);
        set_output("cond", std::move(obj));
    }
};

ZENDEFNODE(RunBeforeFrame, {
    {},
    {"cond"},
    {},
    {"runtime"},
});


struct SetFrameTime : zen::INode {
    virtual void apply() override {
        auto time = get_input<zen::NumericObject>("time")->get<float>();
        zen::state.frame_time = time;
    }
};

ZENDEFNODE(SetFrameTime, {
    {"time"},
    {},
    {},
    {"runtime"},
});

struct GetFrameTime : zen::INode {
    virtual void apply() override {
        auto time = std::make_unique<zen::NumericObject>();
        time->set(zen::state.frame_time);
        set_output("time", std::move(time));
    }
};

ZENDEFNODE(GetFrameTime, {
    {},
    {"time"},
    {},
    {"runtime"},
});

struct GetFrameTimeElapsed : zen::INode {
    virtual void apply() override {
        auto time = std::make_unique<zen::NumericObject>();
        time->set(zen::state.frame_time_elapsed);
        set_output("time", std::move(time));
    }
};

ZENDEFNODE(GetFrameTimeElapsed, {
    {},
    {"time"},
    {},
    {"runtime"},
});

struct GetFrameNum : zen::INode {
    virtual void apply() override {
        auto num = std::make_unique<zen::NumericObject>();
        num->set(zen::state.frameid);
        set_output("FrameNum", std::move(num));
    }
};

ZENDEFNODE(GetFrameNum, {
    {},
    {"FrameNum"},
    {},
    {"runtime"},
});

struct GetTime : zen::INode {
    virtual void apply() override {
        auto time = std::make_unique<zen::NumericObject>();
        time->set(zen::state.frameid * zen::state.frame_time
            + zen::state.frame_time_elapsed);
        set_output("time", std::move(time));
    }
};

ZENDEFNODE(GetTime, {
    {},
    {"time"},
    {},
    {"runtime"},
});

struct GetFramePortion : zen::INode {
    virtual void apply() override {
        auto portion = std::make_unique<zen::NumericObject>();
        portion->set(zen::state.frame_time_elapsed / zen::state.frame_time);
        set_output("FramePortion", std::move(portion));
    }
};

ZENDEFNODE(GetFramePortion, {
    {},
    {"FramePortion"},
    {},
    {"runtime"},
});

struct IntegrateFrameTime : zen::INode {
    virtual void apply() override {
        float dt = zen::state.frame_time;
        if (has_input("desired_dt")) {
            dt = get_input<zen::NumericObject>("desired_dt")->get<float>();
            auto min_scale = get_param<float>("min_scale");
            dt = std::max(std::fabs(dt), min_scale * zen::state.frame_time);
        }
        if (zen::state.frame_time_elapsed + dt >= zen::state.frame_time) {
            dt = zen::state.frame_time - zen::state.frame_time_elapsed;
            zen::state.frame_time_elapsed = zen::state.frame_time;
            zen::state.has_frame_completed = true;
        } else {
            zen::state.frame_time_elapsed += dt;
        }
        zen::state.time_step_integrated = true;
        auto ret = std::make_unique<zen::NumericObject>();
        ret->set(dt);
        set_output("actual_dt", ret);
    }
};

ZENDEFNODE(IntegrateFrameTime, {
    {"desired_dt"},
    {"actual_dt"},
    {{"float", "min_scale", "0.0001"}},
    {"runtime"},
});
