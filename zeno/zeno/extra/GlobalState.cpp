#ifdef ZENO_GLOBALSTATE
#include <zeno/extra/GlobalState.h>

namespace zeno {

ZENO_API GlobalState state;

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

ZENO_API void GlobalState::addViewObject(std::shared_ptr<IObject> const &object) {
    view_objects.push_back(object);
}

ZENO_API void GlobalState::clearViewObjects() {
    view_objects.clear();
}

}
#endif
