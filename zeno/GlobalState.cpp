#include <zeno/GlobalState.h>

namespace zeno {

ZENAPI GlobalState state;

ZENAPI bool GlobalState::substepBegin() {
    if (has_substep_executed) {
        if (!time_step_integrated)
            return false;
    }
    if (has_frame_completed)
        return false;
    return true;
}

ZENAPI void GlobalState::substepEnd() {
    substepid++;
    has_substep_executed = true;
}

ZENAPI void GlobalState::setIOPath(const std::string &iopath_) {
    iopath = iopath_;
}

ZENAPI void GlobalState::frameBegin() {
    has_frame_completed = false;
    has_substep_executed = false;
    time_step_integrated = false;
    frame_time_elapsed = 0;
}

ZENAPI void GlobalState::frameEnd() {
    frameid++;
}

}