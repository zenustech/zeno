#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/PolymorphicMap.h>
#include <memory>
#include <vector>
#include <mutex>
#include <map>

namespace zeno {

struct GlobalComm {
    using ViewObjects = PolymorphicMap<std::map<std::string, std::shared_ptr<IObject>>>;

    struct FrameData {
        ViewObjects view_objects;
        bool b_frame_completed = false;
    };
    std::vector<FrameData> m_frames;
    int m_maxPlayFrame = 0;
    mutable std::mutex m_mtx;

    int beginFrameNumber = 0;
    int endFrameNumber = 0;

    ZENO_API void frameRange(int beg, int end);
    ZENO_API void newFrame();
    ZENO_API void finishFrame();
    ZENO_API void addViewObject(std::string const &key, std::shared_ptr<IObject> object);
    ZENO_API int maxPlayFrames();
    ZENO_API void clearState();
    ZENO_API ViewObjects const *getViewObjects(int frameid);
    ZENO_API ViewObjects const &getViewObjects();
    ZENO_API bool isFrameCompleted(int frameid) const;
};

}
