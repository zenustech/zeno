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

    ZENAPI bool substepBegin();
    ZENAPI void substepEnd();
    ZENAPI void frameBegin();
    ZENAPI void frameEnd();
    ZENAPI void setIOPath(std::string const &iopath_);
};

ZENAPI extern GlobalState state;
}
