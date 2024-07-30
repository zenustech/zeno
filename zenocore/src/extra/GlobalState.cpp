#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/utils/logger.h>
#include <zeno/core/GlobalVariable.h>
#include "reflect/core.hpp"
#include "reflect/type.hpp"
#include "reflect/container/any"
#include "reflect/reflection.generated.hpp"


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
    m_working = false;
    frameid = 0;
    substepid = 0;
    frame_time = 1.f / 60.f;
    frame_time_elapsed = 0;
    has_frame_completed = false;
    has_substep_executed = false;
    time_step_integrated = false;
    sessionid++;
    log_debug("entering session id={}", sessionid);
}

ZENO_API float GlobalState::getFrameId() const {
    zeno::reflect::Any frame = getSession().globalVariableManager->getVariable("$F");
    return frame.has_value() ? zeno::reflect::any_cast<float>(frame) : 0;
    //return frameid;
}

ZENO_API void GlobalState::updateFrameId(float frame) {
    //todo: mutex
    getSession().globalVariableManager->updateVariable(GVariable("$F", zeno::reflect::make_any<float>(frame)));
    //frameid = frame;
}

ZENO_API bool GlobalState::is_working() const {
    std::lock_guard lk(mtx);
    return m_working;
}

ZENO_API void GlobalState::set_working(bool working) {
    std::lock_guard lk(mtx);
    m_working = working;
}

ZENO_API void GlobalState::setCalcObjStatus(CalcObjStatus status) {
    m_status = status;
}

}
