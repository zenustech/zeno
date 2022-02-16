#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/utils/logger.h>

namespace zeno {

ZENO_API GlobalState state;

ZENO_API GlobalState::GlobalState() {}
ZENO_API GlobalState::~GlobalState() = default;

ZENO_API bool GlobalState::substepBegin() {
    if (has_substep_executed) {
        if (!time_step_integrated)
            return false;
    }
    if (has_frame_completed)
        return false;
    return true;
}

ZENO_API void GlobalState::substepEnd() {
    substepid++;
    has_substep_executed = true;
}

ZENO_API void GlobalState::frameBegin() {
    has_frame_completed = false;
    has_substep_executed = false;
    time_step_integrated = false;
    frame_time_elapsed = 0;
}

ZENO_API void GlobalState::frameEnd() {
    frameid++;
}

ZENO_API void GlobalState::clearState() {
    frameid = 0;
    frameid = 0;
    substepid = 0;
    frame_time = 0.03f;
    frame_time_elapsed = 0;
    has_frame_completed = false;
    has_substep_executed = false;
    time_step_integrated = false;
}

}
