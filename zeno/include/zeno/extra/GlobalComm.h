#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/PolymorphicMap.h>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <set>

namespace zeno {

struct GlobalComm {
    using ViewObjects = PolymorphicMap<std::map<std::string, std::shared_ptr<IObject>>>;

    enum FRAME_STATE {
        FRAME_UNFINISH,
        FRAME_COMPLETED,
        FRAME_BROKEN
    };

    struct FrameData {
        ViewObjects view_objects;
        FRAME_STATE frame_state = FRAME_UNFINISH;
    };
    std::vector<FrameData> m_frames;
    int m_maxPlayFrame = 0;
    std::set<int> m_inCacheFrames;
    mutable std::mutex m_mtx;

    int beginFrameNumber = 0;
    int endFrameNumber = 0;
    int maxCachedFrames = 1;
    std::string cacheFramePath;

    ZENO_API void frameCache(std::string const &path, int gcmax);
    ZENO_API void initFrameRange(int beg, int end);
    ZENO_API void newFrame();
    ZENO_API void finishFrame();
    ZENO_API void dumpFrameCache(int frameid, bool cacheLightCameraOnly = false, bool cacheMaterialOnly = false);
    ZENO_API void addViewObject(std::string const &key, std::shared_ptr<IObject> object);
    ZENO_API int maxPlayFrames();
    ZENO_API int numOfFinishedFrame();
    ZENO_API int numOfInitializedFrame();
    ZENO_API std::pair<int, int> frameRange();
    ZENO_API void clearState();
    ZENO_API void clearFrameState();
    ZENO_API ViewObjects const *getViewObjects(const int frameid);
    ZENO_API ViewObjects const &getViewObjects();
    ZENO_API bool isFrameCompleted(int frameid) const;
    ZENO_API FRAME_STATE getFrameState(int frameid) const;
    ZENO_API bool isFrameBroken(int frameid) const;
    ZENO_API int maxCachedFramesNum();
    ZENO_API std::string cachePath();
    ZENO_API bool removeCache(int frame);
};

}
