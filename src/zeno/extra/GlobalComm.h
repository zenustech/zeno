#pragma once

#include <zeno/core/IObject.h>
#include <memory>
#include <vector>
#include <mutex>

namespace zeno {

struct GlobalState;

struct GlobalComm {
    GlobalState *const m_state;

    struct FrameData {
        std::vector<std::shared_ptr<IObject>> view_objects;
    };
    std::vector<FrameData> frames;
    std::mutex mtx;

    ZENO_API void onFrameBegin();
    ZENO_API void addViewObject(std::shared_ptr<IObject> const &object);
    ZENO_API int countFrames();
    ZENO_API void clearState();
    ZENO_API std::vector<std::shared_ptr<IObject>> getViewObjects(int frameid);
};

}
