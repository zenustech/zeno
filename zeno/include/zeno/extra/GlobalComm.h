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

    struct FrameData {
        ViewObjects view_objects;
        bool b_frame_completed = false;
    };
    std::vector<FrameData> m_frames;
    int m_maxPlayFrame = 0;
    std::set<int> m_inCacheFrames;
    mutable std::mutex m_mtx;

    int beginFrameNumber = 0;
    int endFrameNumber = 0;
    int maxCachedFrames = 1;
    std::string cacheFramePath;

    bool isTempDir = false;
    bool cacheautorm = false;

    ZENO_API void frameCache(std::string const &path, int gcmax);
    ZENO_API void initFrameRange(int beg, int end);
    ZENO_API void newFrame();
    ZENO_API void finishFrame();
    ZENO_API void dumpFrameCache(int frameid, bool cacheLightCameraOnly = false, bool cacheMaterialOnly = false);
    ZENO_API void addViewObject(std::string const &key, std::shared_ptr<IObject> object);
    ZENO_API int maxPlayFrames();
    ZENO_API int numOfFinishedFrame();
    ZENO_API std::pair<int, int> frameRange();
    ZENO_API void clearState();
    ZENO_API ViewObjects const *getViewObjects(const int frameid);
    ZENO_API ViewObjects const &getViewObjects();
    ZENO_API bool isFrameCompleted(int frameid) const;

    ZENO_API void setTempDirEnable(bool enable);
    ZENO_API bool tempDirEnabled();
    ZENO_API void setCacheAutoRmEnable(bool enable);
    ZENO_API bool cacheAutoRmEnabled();
};

}
