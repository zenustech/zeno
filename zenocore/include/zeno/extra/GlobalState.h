#pragma once

#include <zeno/utils/api.h>
#include <string>
#include <memory>
#include <vector>
#include <mutex>

namespace zeno {

struct IObject;

enum CalcObjStatus
{
    Collecting,
    Loading,
    Finished,
};

struct GlobalState {

    int substepid = 0;
    float frame_time = 1.f / 60.f;
    float frame_time_elapsed = 0;
    bool has_frame_completed = false;
    bool has_substep_executed = false;
    bool time_step_integrated = false;
    int sessionid = 0;
    std::string zeno_version;

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

    inline bool isFirstSubstep() const {
        return substepid == 0;
    }

    ZENO_API GlobalState();
    ZENO_API ~GlobalState();

    ZENO_API bool substepBegin();
    ZENO_API void substepEnd();
    ZENO_API void frameBegin();
    ZENO_API void frameEnd();
    ZENO_API void clearState();
    ZENO_API float getFrameId() const;
    ZENO_API void updateFrameId(float frameid);
    ZENO_API CalcObjStatus getCalcObjStatus() const { return m_status; }
    ZENO_API void setCalcObjStatus(CalcObjStatus status);
    ZENO_API void set_working(bool working);
    ZENO_API bool is_working() const;

private:
    int frameid = 0;
    bool m_working = false;
    CalcObjStatus m_status = Finished;
    mutable std::mutex mtx;
};

}
