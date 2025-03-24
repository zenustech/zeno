#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/PolymorphicMap.h>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <set>
#include <functional>

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
    std::map<int, std::map<std::string, std::tuple<std::string, int, int, std::string, std::string>>> m_inCacheFrames;//<帧号,该帧的stampinfo:<objkey, tuple<changinfo,baseframe,objtype, fullobjkey>>
    int currentFrameNumber = 0;
    mutable std::mutex m_mtx;

    int beginFrameNumber = 0;
    int endFrameNumber = 0;
    int maxCachedFrames = 1;
    std::string cacheFramePath;
    std::string objTmpCachePath;

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
    ZENO_API bool load_objects(const int frameid, 
                const std::function<bool(std::map<std::string, std::shared_ptr<zeno::IObject>> const& objs)>& cb,
                bool& isFrameValid);
    ZENO_API void clear_objects(const std::function<void()>& cb);
    ZENO_API bool isFrameCompleted(int frameid) const;
    ZENO_API FRAME_STATE getFrameState(int frameid) const;
    ZENO_API bool isFrameBroken(int frameid) const;
    ZENO_API int maxCachedFramesNum();
    ZENO_API std::string cachePath();
    ZENO_API bool removeCache(int frame);
    ZENO_API void removeCachePath();
    static void toDisk(std::string cachedir, int frameid, GlobalComm::ViewObjects& objs, bool cacheLightCameraOnly, bool cacheMaterialOnly, std::string fileName = "", bool isBeginframe = true);
    static bool fromDisk(std::string cachedir, int frameid, GlobalComm::ViewObjects& objs, std::string fileName = "");

    //stamp相关
    static int getObjType(std::shared_ptr<IObject> obj);
    static std::shared_ptr<IObject> constructEmptyObj(int type);
    bool fromDiskByStampinfo(std::string cachedir, int frameid, GlobalComm::ViewObjects& objs, std::map<std::string, std::tuple<std::string, int, int, std::string, std::string>>& newFrameStampInfo);
private:
    ViewObjects const *_getViewObjects(const int frameid);
};

}
