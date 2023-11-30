#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/utils/log.h>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <zeno/types/UserData.h>
#include <unordered_set>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/CameraObject.h>
#include <zeno/types/ListObject.h>
#ifdef __linux__
    #include<unistd.h>
    #include <sys/statfs.h>
#endif
#define MIN_DISKSPACE_MB 1024

namespace zeno {

void GlobalComm::toDisk(std::string cachedir, int frameid, GlobalComm::ViewObjects &objs, std::string key, bool dumpCacheVersionInfo) {
    if (cachedir.empty()) return;

    std::filesystem::path dir = std::filesystem::u8path(cachedir + "/" + std::to_string(1000000 + frameid).substr(1));
    if (!std::filesystem::exists(dir) && !std::filesystem::create_directories(dir))
    {
        log_critical("can not create path: {}", dir);
    }
    if (dumpCacheVersionInfo)
    {
        std::map<std::string, std::string> toViewNodesInfo;

        std::filesystem::path toViewInfoPath = dir / "toViewInofs.zencache";
        if (std::filesystem::exists(toViewInfoPath))
        {
            auto szBuffer = std::filesystem::file_size(toViewInfoPath);
            if (szBuffer != 0)
            {
                std::vector<char> dat(szBuffer);
                FILE* fp = fopen(toViewInfoPath.string().c_str(), "rb");
                if (!fp) {
                    log_error("zeno cache file does not exist");
                    return;
                }
                size_t ret = fread(&dat[0], 1, szBuffer, fp);
                assert(ret == szBuffer);
                fclose(fp);
                fp = nullptr;

                size_t beginpos = 0;
                size_t keyLen = 0;
                std::vector<char>::iterator beginIterator = dat.begin();
                for (auto i = dat.begin(); i != dat.end(); i++)
                {
                    if (*i == '\a')
                    {
                        keyLen = i - beginIterator;
                        std::string key(dat.data() + beginpos, keyLen);
                        toViewNodesInfo.insert(std::make_pair(std::move(key.substr(0, key.find(":"))), std::move(key)));
                        beginpos += i - beginIterator + 1;
                        beginIterator = i + 1;
                    }
                }
            }
        }
        for (auto const& [key, obj] : objs) {
            if (toViewNodesInfo.count(key.substr(0, key.find(":")))) {
                toViewNodesInfo[key.substr(0, key.find(":"))] = key;
            }
            toViewNodesInfo.insert(std::make_pair(key.substr(0, key.find(":")), key));
        }
        std::string keys;
        for (auto const& [id, key] : toViewNodesInfo) {
            keys.append(key);
            keys.push_back('\a');
        }
        std::ofstream ofs(toViewInfoPath, std::ios::binary);
        std::ostreambuf_iterator<char> oit(ofs);
        std::copy(keys.begin(), keys.end(), oit);
    }
    else {
        std::filesystem::path cachepath = dir / (key + ".zencache");
        std::vector<char> bufCaches;
        std::vector<size_t> poses;
        std::string keys;
        for (auto const& [key, obj] : objs) {
            size_t bufsize = bufCaches.size();

            std::back_insert_iterator<std::vector<char>> it(bufCaches);
            if (encodeObject(obj.get(), bufCaches))
            {
                keys.push_back('\a');
                keys.append(key);
                poses.push_back(bufsize);
            }
        }
        keys.push_back('\a');
        keys = "ZENCACHE" + std::to_string(poses.size()) + keys;
        poses.push_back(bufCaches.size());
        size_t currentFrameSize = keys.size() + poses.size() * sizeof(size_t) + bufCaches.size();

        size_t freeSpace = 0;
#ifdef __linux__
        struct statfs diskInfo;
        statfs(std::filesystem::u8path(cachedir).c_str(), &diskInfo);
        freeSpace = diskInfo.f_bsize * diskInfo.f_bavail;
#else
        freeSpace = std::filesystem::space(std::filesystem::u8path(cachedir)).free;
#endif
        //wait in two case: 1. available space minus current frame size less than 1024MB, 2. available space less or equal than 1024MB
        while (((freeSpace >> 20) - MIN_DISKSPACE_MB) < (currentFrameSize >> 20) || (freeSpace >> 20) <= MIN_DISKSPACE_MB)
        {
#ifdef __linux__
            zeno::log_critical("Disk space almost full on {}, wait for zencache remove", std::filesystem::u8path(cachedir).string());
            sleep(2);
            statfs(std::filesystem::u8path(cachedir).c_str(), &diskInfo);
            freeSpace = diskInfo.f_bsize * diskInfo.f_bavail;

#else
            zeno::log_critical("Disk space almost full on {}, wait for zencache remove", std::filesystem::u8path(cachedir).root_path().string());
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            freeSpace = std::filesystem::space(std::filesystem::u8path(cachedir)).free;
#endif
        }

        log_debug("dump cache to disk {}", cachepath);
        std::ofstream ofs(cachepath, std::ios::binary);
        std::ostreambuf_iterator<char> oit(ofs);
        std::copy(keys.begin(), keys.end(), oit);
        std::copy_n((const char*)poses.data(), poses.size() * sizeof(size_t), oit);
        std::copy(bufCaches.begin(), bufCaches.end(), oit);
    }
    
    objs.clear();
}

bool GlobalComm::fromDiskByObjsManager(std::string cachedir, int frameid, GlobalComm::ViewObjects& objs, std::vector<std::string>& nodesToLoad)
{
    if (cachedir.empty())
        return false;
    objs.clear();
    auto dir = std::filesystem::u8path(cachedir) / std::to_string(1000000 + frameid).substr(1);
    if (!std::filesystem::exists(dir))
        return false;

    std::map<std::string, std::string> toViewNodesInfo;

    std::filesystem::path filePath = dir / "toViewInofs.zencache";
    if (!std::filesystem::is_directory(filePath) && std::filesystem::exists(filePath)) {
        auto szBuffer = std::filesystem::file_size(filePath);
        if (szBuffer == 0)
            return true;

        std::vector<char> dat(szBuffer);
        FILE* fp = fopen(filePath.string().c_str(), "rb");
        if (!fp) {
            log_error("zeno cache file does not exist");
            return false;
        }
        size_t ret = fread(&dat[0], 1, szBuffer, fp);
        assert(ret == szBuffer);
        fclose(fp);
        fp = nullptr;

        size_t beginpos = 0;
        size_t keyLen = 0;
        std::vector<char>::iterator beginIterator = dat.begin();
        for (auto i = dat.begin(); i != dat.end(); i++)
        {
            if (*i == '\a')
            {
                keyLen = i - beginIterator;
                std::string key(dat.data() + beginpos, keyLen);
                toViewNodesInfo.insert(std::make_pair(std::move(key.substr(0, key.find(":"))), std::move(key)));
                beginpos += i - beginIterator + 1;
                beginIterator = i + 1;
            }
        }
    }

    std::function<void(zany const&, std::string, std::string)> convertToView = [&](zany const& p, std::string postfix, std::string name) -> void {
        if (ListObject* lst = dynamic_cast<ListObject*>(p.get())) {
            log_info("ToView got ListObject (size={}), expanding", lst->arr.size());
            for (size_t i = 0; i < lst->arr.size(); i++) {
                zany const& lp = lst->arr[i];
                std::string id(name);
                convertToView(lp, postfix + ":LIST" + std::to_string(i), id.insert(id.find(":"), postfix + ":LIST:" + std::to_string(i)));
            }
            return;
        }
        if (!p) {
            log_error("ToView: given object is nullptr");
        }
        else {
            objs.try_emplace(name, std::move(p));
        }
    };
    for (auto& cache : nodesToLoad)
    {
        if (toViewNodesInfo.find(cache) == toViewNodesInfo.end())
            continue;
        std::string toViewObjInfo = toViewNodesInfo[cache];

        std::filesystem::path cachePath = dir / (cache + ".zencache");
        if (!std::filesystem::is_directory(cachePath) && std::filesystem::exists(cachePath)) {
            auto szBuffer = std::filesystem::file_size(cachePath);
            if (szBuffer == 0)
                return true;

            log_debug("load cache from disk {}", cachePath);

            std::vector<char> dat(szBuffer);
            FILE* fp = fopen(cachePath.string().c_str(), "rb");
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
            std::copy_n(dat.data() + pos, (keyscount + 1) * sizeof(size_t), (char*)poses.data());
            pos += (keyscount + 1) * sizeof(size_t);

            int lastObjIdx = keyscount - 1; //now only first output is needed to view this obj.
            if (poses[lastObjIdx] > dat.size() - pos || poses[lastObjIdx + 1] < poses[lastObjIdx]) {
                log_error("zeno cache file broken (4.{})", lastObjIdx);
            }
            const char* p = dat.data() + pos + poses[lastObjIdx];

            convertToView(decodeObject(p, poses[lastObjIdx + 1] - poses[lastObjIdx]), {}, toViewObjInfo.insert(toViewObjInfo.find(":"), ":TOVIEW"));
        }
    }
    return true;
}

bool GlobalComm::fromDiskByRunner(std::string cachedir, int frameid, GlobalComm::ViewObjects &objs, std::string filename) {
    if (cachedir.empty())
        return false;
    objs.clear();
    auto dir = std::filesystem::u8path(cachedir) / std::to_string(1000000 + frameid).substr(1);
    if (!std::filesystem::exists(dir))
        return false;

    std::filesystem::path filePath = dir / (filename + ".zencache");
    if (!std::filesystem::is_directory(filePath) && std::filesystem::exists(filePath)) {

        auto szBuffer = std::filesystem::file_size(filePath);
        if (szBuffer == 0)
            return true;

        log_debug("load cache from disk {}", filePath);

        std::vector<char> dat(szBuffer);
        FILE* fp = fopen(filePath.string().c_str(), "rb");
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
        std::copy_n(dat.data() + pos, (keyscount + 1) * sizeof(size_t), (char*)poses.data());
        pos += (keyscount + 1) * sizeof(size_t);
        for (int k = 0; k < keyscount; k++) {
            if (poses[k] > dat.size() - pos || poses[k + 1] < poses[k]) {
                log_error("zeno cache file broken (4.{})", k);
            }
            const char* p = dat.data() + pos + poses[k];
            objs.try_emplace(keys[k], decodeObject(p, poses[k + 1] - poses[k]));
        }
    }
    return true;
}

ZENO_API void GlobalComm::newFrame() {
    std::lock_guard lck(m_mtx);
    log_debug("GlobalComm::newFrame {}", m_frames.size());
    m_frames.emplace_back().frame_state = FRAME_UNFINISH;
}

ZENO_API void GlobalComm::finishFrame() {
    std::lock_guard lck(m_mtx);
    log_debug("GlobalComm::finishFrame {}", m_maxPlayFrame);
    if (m_maxPlayFrame >= 0 && m_maxPlayFrame < m_frames.size())
        m_frames[m_maxPlayFrame].frame_state = FRAME_COMPLETED;
    m_maxPlayFrame += 1;
}

ZENO_API void GlobalComm::dumpFrameCache(int frameid) {
    std::lock_guard lck(m_mtx);
    int frameIdx = frameid - beginFrameNumber;
    if (frameIdx >= 0 && frameIdx < m_frames.size()) {
        log_debug("dumping frame {}", frameid);
        toDisk(cacheFramePath, frameid, m_frames[frameIdx].view_objects, "", true);
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

ZENO_API void GlobalComm::clearFrameState()
{
    std::lock_guard lck(m_mtx);
    m_frames.clear();
    m_inCacheFrames.clear();
    m_maxPlayFrame = 0;
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

ZENO_API int GlobalComm::numOfInitializedFrame()
{
    std::lock_guard lck(m_mtx);
    return m_frames.size();
}

ZENO_API std::pair<int, int> GlobalComm::frameRange() {
    std::lock_guard lck(m_mtx);
    return std::pair<int, int>(beginFrameNumber, endFrameNumber);
}

ZENO_API GlobalComm::ViewObjects const *GlobalComm::getViewObjects(const int frameid) {
    std::lock_guard lck(m_mtx);
    return _getViewObjects(frameid);
}

GlobalComm::ViewObjects const* GlobalComm::_getViewObjects(const int frameid) {
    int frameIdx = frameid - beginFrameNumber;
    if (frameIdx < 0 || frameIdx >= m_frames.size())
        return nullptr;
    if (maxCachedFrames != 0) {
        // load back one gc:
        if (!m_inCacheFrames.count(frameid)) {  // notinmem then cacheit
            bool ret = fromDiskByObjsManager(cacheFramePath, frameid, m_frames[frameIdx].view_objects, toViewNodesId);
            if (!ret)
                return nullptr;

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

ZENO_API bool GlobalComm::load_objects(
        const int frameid,
        const std::function<bool(std::map<std::string, std::shared_ptr<zeno::IObject>> const& objs)>& callback,
        bool& isFrameValid)
{
    if (!callback)
        return false;

    std::lock_guard lck(m_mtx);

    int frame = frameid;
    frame -= beginFrameNumber;
    if (frame < 0 || frame >= m_frames.size() || m_frames[frame].frame_state != FRAME_COMPLETED)
    {
        isFrameValid = false;
        return false;
    }

    isFrameValid = true;
    bool inserted = false;
    auto const* viewObjs = _getViewObjects(frameid);
    if (viewObjs) {
        zeno::log_trace("load_objects: {} objects at frame {}", viewObjs->size(), frameid);
        inserted = callback(viewObjs->m_curr);
    }
    else {
        zeno::log_trace("load_objects: no objects at frame {}", frameid);
        inserted = callback({});
    }
    return inserted;
}

ZENO_API bool GlobalComm::isFrameCompleted(int frameid) const {
    std::lock_guard lck(m_mtx);
    frameid -= beginFrameNumber;
    if (frameid < 0 || frameid >= m_frames.size())
        return false;
    return m_frames[frameid].frame_state == FRAME_COMPLETED;
}

ZENO_API GlobalComm::FRAME_STATE GlobalComm::getFrameState(int frameid) const
{
    std::lock_guard lck(m_mtx);
    frameid -= beginFrameNumber;
    if (frameid < 0 || frameid >= m_frames.size())
        return FRAME_UNFINISH;
    return m_frames[frameid].frame_state;
}

ZENO_API bool GlobalComm::isFrameBroken(int frameid) const
{
    std::lock_guard lck(m_mtx);
    frameid -= beginFrameNumber;
    if (frameid < 0 || frameid >= m_frames.size())
        return false;
    return m_frames[frameid].frame_state == FRAME_BROKEN;
}

ZENO_API int GlobalComm::maxCachedFramesNum()
{
    std::lock_guard lck(m_mtx);
    return maxCachedFrames;
}

ZENO_API std::string GlobalComm::cachePath()
{
    std::lock_guard lck(m_mtx);
    return cacheFramePath;
}

ZENO_API bool GlobalComm::removeCache(int frame)
{
    std::lock_guard lck(m_mtx);
    bool hasZencacheOnly = true;
    std::filesystem::path dirToRemove = std::filesystem::u8path(cacheFramePath + "/" + std::to_string(1000000 + frame).substr(1));
    if (std::filesystem::exists(dirToRemove))
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
            m_frames[frame - beginFrameNumber].frame_state = FRAME_BROKEN;
            std::filesystem::remove_all(dirToRemove);
            zeno::log_info("remove dir: {}", dirToRemove);
        }
    }
    if (frame == endFrameNumber && std::filesystem::exists(std::filesystem::u8path(cacheFramePath)) && std::filesystem::is_empty(std::filesystem::u8path(cacheFramePath)))
    {
        std::filesystem::remove(std::filesystem::u8path(cacheFramePath));
        zeno::log_info("remove dir: {}", std::filesystem::u8path(cacheFramePath).string());
    }
    return true;
}

ZENO_API void GlobalComm::removeCachePath()
{
    std::lock_guard lck(m_mtx);
    std::filesystem::path dirToRemove = std::filesystem::u8path(cacheFramePath);
    if (std::filesystem::exists(dirToRemove) && cacheFramePath.find(".") == std::string::npos)
    {
        std::filesystem::remove_all(dirToRemove);
        zeno::log_info("remove dir: {}", dirToRemove);
    }
}

ZENO_API void GlobalComm::setToViewNodes(std::vector<std::string>&nodes)
{
    std::lock_guard lck(m_mtx);
    toViewNodesId = std::move(nodes);
}

}
