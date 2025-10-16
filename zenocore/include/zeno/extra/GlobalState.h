#pragma once

#include <zeno/utils/api.h>
#include <string>
#include <memory>
#include <vector>
#include <mutex>
#include <set>
#include <map>
#include <zeno/core/common.h>

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
    ZENO_API void updateFrameRange(int start, int end);
    ZENO_API int getStartFrame() const;
    ZENO_API int getEndFrame() const;
    ZENO_API CalcObjStatus getCalcObjStatus() const { return m_status; }
    ZENO_API void setCalcObjStatus(CalcObjStatus status);
    ZENO_API void set_working(bool working);
    ZENO_API bool is_working() const;

    ZENO_API void init_total_runtime(float total_time);
    ZENO_API void update_consume_time(float t);
    ZENO_API float get_total_runtime() const;
    ZENO_API float get_consume_time() const;

    ZENO_API void inc_io_processed(int inc);
    ZENO_API int get_io_processed() const;

private:
    int frameid = 0;
    int frame_start = 0;
    int frame_end = 0;
    bool m_working = false;
    CalcObjStatus m_status = Finished;
    float total_time = 0.f;
    float time_consumed = 0.f;
    float total_io_units = 0.f;
    float processed_io_units = 0.f;

    mutable std::mutex mtx;
};

}
