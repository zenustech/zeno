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
    };
    std::vector<FrameData> m_frames;
    int m_maxPlayFrame = 0;
    mutable std::mutex m_mtx;

    ZENO_API void newFrame();
    ZENO_API void finishFrame();
    ZENO_API void addViewObject(std::string const &key, std::shared_ptr<IObject> object);
    ZENO_API int maxPlayFrames();
    ZENO_API void clearState();
    ZENO_API ViewObjects const &getViewObjects(int frameid);
    ZENO_API ViewObjects const &getViewObjects();
};

}
