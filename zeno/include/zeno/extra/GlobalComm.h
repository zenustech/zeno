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
#include <filesystem>

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
    std::map<int, std::map<std::string, std::tuple<std::string, int, int, std::string, std::string, size_t, size_t>>> m_inCacheFrames;//<帧号,该帧的stampinfo:<objkey, tuple<changinfo,baseframe,objtype, fullobjkey>>
    int currentFrameNumber = 0;
    mutable std::mutex m_mtx;

    int beginFrameNumber = 0;
    int endFrameNumber = 0;
    int maxCachedFrames = 1;
    std::string cacheFramePath;
    std::string objTmpCachePath;

    std::map<uintptr_t, std::tuple<bool, bool, bool>> sceneLoadedFlag;  //assetneedLoad, run, load
    bool assetsInitialized = false;

    ZENO_API void frameCache(std::string const &path, int gcmax);
    ZENO_API void initFrameRange(int beg, int end);
    ZENO_API void newFrame();
    ZENO_API void finishFrame();
    ZENO_API void dumpFrameCache(int frameid, std::string runtype = "RunAll");
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
                const std::function<bool(std::map<std::string, std::shared_ptr<zeno::IObject>> const& objs, std::string& runtype)>& cb,
                std::function<void(int frameid, bool inserted, bool hasstamp)> callbackUpdate, uintptr_t sceneId,
                bool& isFrameValid);
    ZENO_API void clear_objects(const std::function<void()>& cb);
    ZENO_API bool isFrameCompleted(int frameid) const;
    ZENO_API FRAME_STATE getFrameState(int frameid) const;
    ZENO_API bool isFrameBroken(int frameid) const;
    ZENO_API int maxCachedFramesNum();
    ZENO_API std::string cachePath();
    ZENO_API bool removeCache(int frame);
    ZENO_API void removeCachePath();
    ZENO_API std::string cacheTimeStamp(int frame, bool& exists);
    void toDisk(std::string cachedir, int frameid, GlobalComm::ViewObjects& objs, std::string runtype, std::string fileName = "", bool isBeginframe = true);
    bool fromDisk(std::string cachedir, int frameid, GlobalComm::ViewObjects& objs, std::string& runtype, std::string fileName = "");

    //stamp相关
    static int getObjType(std::shared_ptr<IObject> obj);
    static std::shared_ptr<IObject> constructEmptyObj(int type);
    bool fromDiskByStampinfo(std::string cachedir, int frameid, GlobalComm::ViewObjects& objs, std::map<std::string, std::tuple<std::string, int, int, std::string, std::string, size_t, size_t>>& newFrameStampInfo, std::string& runtype);
    std::shared_ptr<IObject> fromDiskReadObject(std::string cachedir, int frameid, std::string objectName);
    static std::string getRunType(std::filesystem::path dir);
private:
    ViewObjects const *_getViewObjects(const int frameidm, uintptr_t sceneIdn, std::string& runtype, bool& hasStamp);
};

}
