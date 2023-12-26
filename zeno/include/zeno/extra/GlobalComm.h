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
#include <zeno/extra/ObjectsManager.h>
//-----ObjectsManager-----
#include <zeno/utils/MapStablizer.h>
#include <zeno/utils/disable_copy.h>
#include <optional>
#include "zeno/utils/vec.h"

namespace zeno {

    extern ZENO_API std::recursive_mutex g_objsMutex;
    using MapObjects = std::map<std::string, std::shared_ptr<zeno::IObject>>;

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
    ViewObjects m_static_objects;

    int m_maxPlayFrame = 0;
    std::set<int> m_inCacheFrames;

    int beginFrameNumber = 0;
    int endFrameNumber = 0;
    int maxCachedFrames = 1;
    std::string cacheFramePath;
    std::string objTmpCachePath;

    ZENO_API void frameCache(std::string const &path, int gcmax);
    ZENO_API void initFrameRange(int beg, int end);
    ZENO_API void newFrame();
    ZENO_API void finishFrame();
    ZENO_API void dumpFrameCache(int frameid);
    ZENO_API void addViewObject(std::string const &key, std::shared_ptr<IObject> object);
    ZENO_API void addStaticObject(std::string const& key, std::shared_ptr<IObject> object);
    ZENO_API int maxPlayFrames();
    ZENO_API int numOfFinishedFrame();
    ZENO_API int numOfInitializedFrame();
    ZENO_API std::pair<int, int> frameRange();
    ZENO_API void clearState();
    ZENO_API void clearFrameState();
    ZENO_API std::shared_ptr<IObject> getViewObject(std::string const& key);
    ZENO_API bool load_objects(const int frameid, bool& isFrameValid);
    ZENO_API bool isFrameCompleted(int frameid) const;
    ZENO_API FRAME_STATE getFrameState(int frameid) const;
    ZENO_API bool isFrameBroken(int frameid) const;
    ZENO_API int maxCachedFramesNum();
    ZENO_API std::string cachePath();
    ZENO_API bool removeCache(int frame);
    ZENO_API void removeCachePath();
    ZENO_API void setToViewNodes(std::vector<std::string>& nodes);
    static void toDisk(std::string cachedir, int frameid, GlobalComm::ViewObjects& objs, std::string key = "", bool dumpCacheVersionInfo = false);
    static bool fromDiskByRunner(std::string cachedir, int frameid, GlobalComm::ViewObjects& objs, std::string filename);
    static bool fromDiskByObjsManager(std::string cachedir, int frameid, GlobalComm::ViewObjects& objs, std::vector<std::string>& nodesToLoad);

    //-----ObjectsManager-----
    ZENO_API void clear_objects();
    ZENO_API std::optional<zeno::IObject*> get(std::string nid);
    enum RenderType
    {
        NORMAL = 0,
        LIGHT_CAMERA,
        MATERIAL,
        UNDEFINED,
    };

    ZENO_API std::vector<std::pair<std::string, IObject*>> pairs() const;
    ZENO_API std::vector<std::pair<std::string, std::shared_ptr<IObject>>> pairsShared() const;
    //------new change------
    ZENO_API void clear_lightObjects();
    ZENO_API bool lightObjsCount(std::string& id);
    ZENO_API bool objsCount(std::string& id);
    ZENO_API const std::string getLightObjKeyByLightObjID(std::string id);
    ZENO_API const std::string getObjKeyByObjID(std::string& id);
    ZENO_API bool getLightObjData(std::string& id, zeno::vec3f& pos, zeno::vec3f& scale, zeno::vec3f& rotate, zeno::vec3f& clr, float& intensity);
    ZENO_API bool setLightObjData(std::string& id, zeno::vec3f& pos, zeno::vec3f& scale, zeno::vec3f& rotate, zeno::vec3f& rgb, float& intensity, std::vector<zeno::vec3f>& verts);
    ZENO_API bool setProceduralSkyData(std::string id, zeno::vec2f& sunLightDir, float& sunSoftnessValue, zeno::vec2f& windDir, float& timeStartValue, float& timeSpeedValue, float& sunLightIntensityValue, float& colorTemperatureMixValue, float& colorTemperatureValue);
    ZENO_API bool getProceduralSkyData(std::string& id, zeno::vec2f& sunLightDir, float& sunSoftnessValue, zeno::vec2f& windDir, float& timeStartValue, float& timeSpeedValue, float& sunLightIntensityValue, float& colorTemperatureMixValue, float& colorTemperatureValue);
    ZENO_API void getAllLightsKey(std::vector<std::string>& keys);
    ZENO_API std::string getObjMatId(std::string& id);
    ZENO_API const std::string getObjKey1(std::string& id, int frame);
    ZENO_API MapObjects getLightObjs();

    ZENO_API void addTransferObj(std::string const& key, std::shared_ptr<IObject>);
    ZENO_API MapObjects getTransferObjs();
    ZENO_API void clearTransferObjs();

    ZENO_API int getLightObjsSize();
    ZENO_API bool getNeedUpdateLight();
    ZENO_API void setNeedUpdateLight(bool update);

    ZENO_API RenderType getRenderTypeByObjects(std::map<std::string, std::shared_ptr<zeno::IObject>>& objs);
    ZENO_API void setRenderType(RenderType type);
    ZENO_API RenderType getRenderType();

private:
    ViewObjects const* _getViewObjects(const int frameid);
    void _initStaticObjects();
    std::vector<std::string> toViewNodesId;

    //-----ObjectsManager-----
    MapObjects m_transferObjs;
    MapObjects m_lightObjects;
    bool needUpdateLight = true;

    RenderType renderType = UNDEFINED;
    std::map<std::string, int> lastToViewNodesType;
    //------new change------
    void prepareForOptix(bool inserted, std::map<std::string, std::shared_ptr<zeno::IObject>> const& objs);
    int m_currentFrame = 0;    //ºı»•startFrame
};

}
