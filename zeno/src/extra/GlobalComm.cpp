#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/utils/log.h>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <zeno/types/UserData.h>
#include <unordered_set>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/CameraObject.h>
#ifdef __linux__
    #include<unistd.h>
    #include <sys/statfs.h>
#endif

namespace zeno {

std::vector<std::filesystem::path> cachepath(3);
std::unordered_set<std::string> lightCameraNodes({
    "CameraEval", "CameraNode", "CihouMayaCameraFov", "ExtractCameraData", "GetAlembicCamera","MakeCamera",
    "LightNode", "BindLight", "ProceduralSky", "HDRSky",
    });
std::string matlNode = "ShaderFinalize";

static void toDisk(std::string cachedir, int frameid, GlobalComm::ViewObjects &objs, bool cacheLightCameraOnly, bool cacheMaterialOnly) {
    if (cachedir.empty()) return;
    std::string dir = cachedir + "/" + std::to_string(1000000 + frameid).substr(1);
    if (!std::filesystem::exists(dir) && !std::filesystem::create_directories(dir))
    {
        log_critical("can not create path: {}", dir);
    }
    std::vector<std::vector<char>> bufCaches(3);
    std::vector<std::vector<size_t>> poses(3);
    std::vector<std::string> keys(3);
    for (auto const &[key, obj]: objs) {

        size_t bufsize =0;
        std::string nodeName = key.substr(key.find("-") + 1, key.find(":") - key.find("-") -1);
        if (cacheLightCameraOnly && (lightCameraNodes.count(nodeName) || obj->userData().get2<int>("isL", 0) || std::dynamic_pointer_cast<CameraObject>(obj)))
        {
            bufsize = bufCaches[0].size();
            if (encodeObject(obj.get(), bufCaches[0]))
            {
                keys[0].push_back('\a');
                keys[0].append(key);
                poses[0].push_back(bufsize);
            }
        }
        if (cacheMaterialOnly && (matlNode == nodeName || std::dynamic_pointer_cast<MaterialObject>(obj)))
        {
            bufsize = bufCaches[1].size();
            if (encodeObject(obj.get(), bufCaches[1]))
            {
                keys[1].push_back('\a');
                keys[1].append(key);
                poses[1].push_back(bufsize);
            }
        }
        if (!cacheLightCameraOnly && !cacheMaterialOnly)
        {
            if (lightCameraNodes.count(nodeName) || obj->userData().get2<int>("isL", 0) || std::dynamic_pointer_cast<CameraObject>(obj)) {
                bufsize = bufCaches[0].size();
                if (encodeObject(obj.get(), bufCaches[0]))
                {
                    keys[0].push_back('\a');
                    keys[0].append(key);
                    poses[0].push_back(bufsize);
                }
            } else if (matlNode == nodeName || std::dynamic_pointer_cast<MaterialObject>(obj)) {
                bufsize = bufCaches[1].size();
                if (encodeObject(obj.get(), bufCaches[1]))
                {
                    keys[1].push_back('\a');
                    keys[1].append(key);
                    poses[1].push_back(bufsize);
                }
            } else {
                bufsize = bufCaches[2].size();
                if (encodeObject(obj.get(), bufCaches[2]))
                {
                    keys[2].push_back('\a');
                    keys[2].append(key);
                    poses[2].push_back(bufsize);
                }
            }
        }
    }
    cachepath[0] = std::filesystem::u8path(dir) / "lightCameraObj.zencache";
    cachepath[1] = std::filesystem::u8path(dir) / "materialObj.zencache";
    cachepath[2] = std::filesystem::u8path(dir) / "normalObj.zencache";
    size_t currentFrameSize = 0;
    for (int i = 0; i < 3; i++)
    {
        if (poses[i].size() == 0)
        {
            continue;
        }
        keys[i].push_back('\a');
        keys[i] = "ZENCACHE" + std::to_string(poses[i].size()) + keys[i];
        poses[i].push_back(bufCaches[i].size());
        currentFrameSize += keys[i].size() + poses[i].size() * sizeof(size_t) + bufCaches[i].size();
    }
    size_t freeSpace = 0;
    #ifdef __linux__
        struct statfs diskInfo;
        statfs(cachedir.c_str(), &diskInfo);
        freeSpace = diskInfo.f_bsize * diskInfo.f_bavail;
    #else
        freeSpace = std::filesystem::space(std::filesystem::u8path(cachedir)).free;
    #endif
    while ((currentFrameSize >> 20) > ((freeSpace >> 20) - 10*1024) || (freeSpace >> 20) <= 10*1024) //Ensure available space 10GB larger than cache
    {
        #ifdef __linux__
            zeno::log_critical("Disk space almost full on {}, wait for zencache remove", std::filesystem::u8path(cachedir).string());
            sleep(2);
            statfs(cachedir.c_str(), &diskInfo);
            freeSpace = diskInfo.f_bsize * diskInfo.f_bavail;

        #else
            zeno::log_critical("Disk space almost full on {}, wait for zencache remove", std::filesystem::u8path(cachedir).root_path().string());
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            freeSpace = std::filesystem::space(std::filesystem::u8path(cachedir)).free;
        #endif
    }
    for (int i = 0; i < 3; i++)
    {
        if (poses[i].size() == 0)
            continue;
        log_critical("dump cache to disk {}", cachepath[i]);
        std::ofstream ofs(cachepath[i], std::ios::binary);
        std::ostreambuf_iterator<char> oit(ofs);
        std::copy(keys[i].begin(), keys[i].end(), oit);
        std::copy_n((const char *)poses[i].data(), poses[i].size() * sizeof(size_t), oit);
        std::copy(bufCaches[i].begin(), bufCaches[i].end(), oit);
    }
    objs.clear();
}

static bool fromDisk(std::string cachedir, int frameid, GlobalComm::ViewObjects &objs) {
    if (cachedir.empty())
        return false;
    objs.clear();
    cachepath[2] = std::filesystem::u8path(cachedir) / std::to_string(1000000 + frameid).substr(1) / "normalObj.zencache";
    cachepath[1] = std::filesystem::u8path(cachedir) / std::to_string(1000000 + frameid).substr(1) / "materialObj.zencache";
    cachepath[0] = std::filesystem::u8path(cachedir) / std::to_string(1000000 + frameid).substr(1) / "lightCameraObj.zencache";
    for (auto path : cachepath)
    {
        if (!std::filesystem::exists(path))
        {
            continue;
        }
        log_critical("load cache from disk {}", path);

        auto szBuffer = std::filesystem::file_size(path);
        std::vector<char> dat(szBuffer);
        FILE *fp = fopen(path.string().c_str(), "rb");
        if (!fp) {
            log_error("zeno cache file does not exist");
            return false;
        }
        size_t ret = fread(&dat[0], 1, szBuffer, fp);
        assert(ret == szBuffer);
        fclose(fp);
        fp = nullptr;

        if (dat.size() <= 8 || std::string(dat.data(), 8) != "ZENCACHE") {
            log_error("zeno cache file broken (1)");
            return false;
        }
        size_t pos = std::find(dat.begin() + 8, dat.end(), '\a') - dat.begin();
        if (pos == dat.size()) {
            log_error("zeno cache file broken (2)");
            return false;
        }
        size_t keyscount = std::stoi(std::string(dat.data() + 8, pos - 8));
        pos = pos + 1;
        std::vector<std::string> keys;
        for (int k = 0; k < keyscount; k++) {
            size_t newpos = std::find(dat.begin() + pos, dat.end(), '\a') - dat.begin();
            if (newpos == dat.size()) {
                log_error("zeno cache file broken (3.{})", k);
                return false;
            }
            keys.emplace_back(dat.data() + pos, newpos - pos);
            pos = newpos + 1;
        }
        std::vector<size_t> poses(keyscount + 1);
        std::copy_n(dat.data() + pos, (keyscount + 1) * sizeof(size_t), (char *)poses.data());
        pos += (keyscount + 1) * sizeof(size_t);
        for (int k = 0; k < keyscount; k++) {
            if (poses[k] > dat.size() - pos || poses[k + 1] < poses[k]) {
                log_error("zeno cache file broken (4.{})", k);
            }
            const char *p = dat.data() + pos + poses[k];
            objs.try_emplace(keys[k], decodeObject(p, poses[k + 1] - poses[k]));
        }
    }
    return true;
}

ZENO_API void GlobalComm::newFrame() {
    std::lock_guard lck(m_mtx);
    log_debug("GlobalComm::newFrame {}", m_frames.size());
    m_frames.emplace_back().b_frame_completed = false;
}

ZENO_API void GlobalComm::finishFrame() {
    std::lock_guard lck(m_mtx);
    log_debug("GlobalComm::finishFrame {}", m_maxPlayFrame);
    if (m_maxPlayFrame >= 0 && m_maxPlayFrame < m_frames.size())
        m_frames[m_maxPlayFrame].b_frame_completed = true;
    m_maxPlayFrame += 1;
}

ZENO_API void GlobalComm::dumpFrameCache(int frameid, bool cacheLightCameraOnly, bool cacheMaterialOnly) {
    std::lock_guard lck(m_mtx);
    int frameIdx = frameid - beginFrameNumber;
    if (frameIdx >= 0 && frameIdx < m_frames.size()) {
        log_debug("dumping frame {}", frameid);
        toDisk(cacheFramePath, frameid, m_frames[frameIdx].view_objects, cacheLightCameraOnly, cacheMaterialOnly);
    }
}

ZENO_API void GlobalComm::addViewObject(std::string const &key, std::shared_ptr<IObject> object) {
    std::lock_guard lck(m_mtx);
    log_debug("GlobalComm::addViewObject {}", m_frames.size());
    if (m_frames.empty()) throw makeError("empty frame cache");
    m_frames.back().view_objects.try_emplace(key, std::move(object));
}

ZENO_API void GlobalComm::clearState() {
    std::lock_guard lck(m_mtx);
    m_frames.clear();
    m_inCacheFrames.clear();
    m_maxPlayFrame = 0;
    maxCachedFrames = 1;
    cacheFramePath = {};
}

ZENO_API void GlobalComm::frameCache(std::string const &path, int gcmax) {
    std::lock_guard lck(m_mtx);
    cacheFramePath = path;
    maxCachedFrames = gcmax;
}

ZENO_API void GlobalComm::initFrameRange(int beg, int end) {
    std::lock_guard lck(m_mtx);
    beginFrameNumber = beg;
    endFrameNumber = end;
}

ZENO_API int GlobalComm::maxPlayFrames() {
    std::lock_guard lck(m_mtx);
    return m_maxPlayFrame + beginFrameNumber; // m_frames.size();
}

ZENO_API int GlobalComm::numOfFinishedFrame() {
    std::lock_guard lck(m_mtx);
    return m_maxPlayFrame;
}

ZENO_API std::pair<int, int> GlobalComm::frameRange() {
    std::lock_guard lck(m_mtx);
    return std::pair<int, int>(beginFrameNumber, endFrameNumber);
}

ZENO_API GlobalComm::ViewObjects const *GlobalComm::getViewObjects(const int frameid) {
    std::lock_guard lck(m_mtx);
    int frameIdx = frameid - beginFrameNumber;
    if (frameIdx < 0 || frameIdx >= m_frames.size())
        return nullptr;
    if (maxCachedFrames != 0) {
        // load back one gc:
        if (!m_inCacheFrames.count(frameid)) {  // notinmem then cacheit
            bool ret = fromDisk(cacheFramePath, frameid, m_frames[frameIdx].view_objects);
            if (!ret)
                return nullptr;

            if (!isTempDir && cacheautorm)
            {
                bool hasZencacheOnly = true;
                std::string dirToRemove = cacheFramePath + "/" + std::to_string(1000000 + frameid).substr(1);
                if (std::filesystem::exists(std::filesystem::u8path(dirToRemove)))
                {
                    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(dirToRemove))
                    {
                        std::string filePath = entry.path().string();
                        if (std::filesystem::is_directory(entry.path()) || filePath.substr(filePath.size() - 9) != ".zencache")
                        {
                            hasZencacheOnly = false;
                            break;
                        }
                    }
                    if (hasZencacheOnly)
                    {
                        std::filesystem::remove_all(dirToRemove);
                        zeno::log_info("remove dir: {}", dirToRemove);
                    }
                }
                if (frameid == endFrameNumber && std::filesystem::exists(cacheFramePath) && std::filesystem::is_empty(cacheFramePath))
                {
                    std::filesystem::remove(cacheFramePath);
                    zeno::log_info("remove dir: {}", cacheFramePath);
                }
            }

            m_inCacheFrames.insert(frameid);
            // and dump one as balance:
            if (m_inCacheFrames.size() && m_inCacheFrames.size() > maxCachedFrames) { // notindisk then dumpit
                for (int i: m_inCacheFrames) {
                    if (i != frameid) {
                        // seems that objs will not be modified when load_objects called later.
                        // so, there is no need to dump.
                        //toDisk(cacheFramePath, i, m_frames[i - beginFrameNumber].view_objects);
                        m_frames[i - beginFrameNumber].view_objects.clear();
                        m_inCacheFrames.erase(i);
                        break;
                    }
                }
            }
        }
    }
    return &m_frames[frameIdx].view_objects;
}

ZENO_API GlobalComm::ViewObjects const &GlobalComm::getViewObjects() {
    std::lock_guard lck(m_mtx);
    return m_frames.back().view_objects;
}

ZENO_API bool GlobalComm::isFrameCompleted(int frameid) const {
    std::lock_guard lck(m_mtx);
    frameid -= beginFrameNumber;
    if (frameid < 0 || frameid >= m_frames.size())
        return false;
    return m_frames[frameid].b_frame_completed;
}

ZENO_API void GlobalComm::setTempDirEnable(bool enable)
{
    std::lock_guard lck(m_mtx);
    isTempDir = enable;
}

ZENO_API bool GlobalComm::tempDirEnabled()
{
    std::lock_guard lck(m_mtx);
    return isTempDir;
}

ZENO_API void GlobalComm::setCacheAutoRmEnable(bool enable)
{
    std::lock_guard lck(m_mtx);
    cacheautorm = enable;
}

ZENO_API bool GlobalComm::cacheAutoRmEnabled()
{
    std::lock_guard lck(m_mtx);
    return cacheautorm;
}

}
