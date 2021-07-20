#pragma once

#include <zeno/zeno.h>

namespace zeno {

struct GlobalState {
    std::string iopath = "/tmp/zenio";
    int frameid = 0;
    int substepid = 0;
    float frame_time = 0.03f;
    float frame_time_elapsed = 0;
    bool has_frame_completed = false;
    bool has_substep_executed = false;
    bool time_step_integrated = false;

    inline bool isAfterFrame() const {
        return has_frame_completed || !time_step_integrated;
    }

    inline bool isBeforeFrame() const {
        return !has_substep_executed;
    }

    inline bool isOneSubstep() const {
        return (time_step_integrated && has_frame_completed)
            || (!has_substep_executed && !time_step_integrated);
    }

    inline bool isFirstFrame() const {
        return substepid == 0;
    }

    ZENAPI bool substepBegin();
    ZENAPI void substepEnd();
    ZENAPI void frameBegin();
    ZENAPI void frameEnd();
    ZENAPI void setIOPath(std::string const &iopath_);
};

ZENAPI extern GlobalState state;
}
