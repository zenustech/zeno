#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/utils/logger.h>
#include <zeno/core/GlobalVariable.h>
#include "reflect/core.hpp"
#include "reflect/type.hpp"
#include "reflect/container/any"
#include "zeno_types/reflect/reflection.generated.hpp"


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
    return frameid;
}

ZENO_API void GlobalState::updateFrameId(float frame) {
    //todo: mutex
    frameid = frame;
}

ZENO_API void GlobalState::updateFrameRange(int start, int end)
{
    getSession().globalVariableManager->updateVariable(GVariable("startFarme", zeno::reflect::make_any<int>(start)));
    getSession().globalVariableManager->updateVariable(GVariable("endFrame", zeno::reflect::make_any<int>(end)));
}

ZENO_API int GlobalState::getStartFrame() const
{
    zeno::reflect::Any start = getSession().globalVariableManager->getVariable("startFarme");
    return start.has_value() ? zeno::reflect::any_cast<int>(start) : 0;
}

ZENO_API int GlobalState::getEndFrame() const
{
    zeno::reflect::Any end = getSession().globalVariableManager->getVariable("endFrame");
    return end.has_value() ? zeno::reflect::any_cast<int>(end) : 0;
}

ZENO_API bool GlobalState::is_working() const {
    std::lock_guard lk(mtx);
    return m_working;
}

ZENO_API void GlobalState::addGlobalVarNode(std::string var, ObjPath nodeUUid)
{
    auto it = globalVarNodesMap.find(var);
    if (it == globalVarNodesMap.end())
    {
        globalVarNodesMap.insert({ var, {nodeUUid} });
    }
    else {
        globalVarNodesMap[var].insert(nodeUUid);
    }
}

ZENO_API void GlobalState::removeGlobalVarNode(std::string var, ObjPath nodeUUid)
{
    auto it = globalVarNodesMap.find(var);
    if (it != globalVarNodesMap.end())
    {
        globalVarNodesMap[var].erase(nodeUUid);
    }
}

ZENO_API std::set<ObjPath> GlobalState::getDenpendentNodes(std::string var)
{
    auto it = globalVarNodesMap.find(var);
    if (it != globalVarNodesMap.end())
    {
        return globalVarNodesMap[var];
    }
    return std::set<std::string> ();
}

ZENO_API void GlobalState::set_working(bool working) {
    std::lock_guard lk(mtx);
    m_working = working;
}

ZENO_API void GlobalState::setCalcObjStatus(CalcObjStatus status) {
    m_status = status;
}

}
