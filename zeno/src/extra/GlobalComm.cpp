#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/utils/log.h>
#include <filesystem>
#include <algorithm>
#include <fstream>

namespace zeno {

static void toDisk(std::string cachedir, int frameid, GlobalComm::ViewObjects &objs) {
    if (cachedir.empty()) return;
    std::vector<char> buf;
    std::vector<size_t> poses;
    std::string keys = "ZENCACHE" + std::to_string((int)objs.size());

    for (auto const &[key, obj]: objs) {
        keys.push_back('\a');
        keys.append(key);
        poses.push_back(buf.size());
        encodeObject(obj.get(), buf);
    }
    poses.push_back(buf.size());
    keys.push_back('\a');

    auto path = std::filesystem::path(cachedir) / (std::to_string(1000000 + frameid).substr(1) + ".zencache");
    log_debug("dump cache to disk {}", path);
    std::ofstream ofs(path);
    std::ostreambuf_iterator<char> oit(ofs);
    std::copy(keys.begin(), keys.end(), oit);
    std::copy_n((const char *)poses.data(), poses.size() * sizeof(size_t), oit);
    std::copy(buf.begin(), buf.end(), oit);
    objs.clear();
}

static void fromDisk(std::string cachedir, int frameid, GlobalComm::ViewObjects &objs) {
    if (cachedir.empty()) return;
    objs.clear();
    auto path = std::filesystem::path(cachedir) / (std::to_string(1000000 + frameid).substr(1) + ".zencache");
    log_debug("load cache from disk {}", path);
    std::ifstream ifs(path);
    std::istreambuf_iterator<char> iit(ifs), iite;
    std::vector<char> dat;
    std::copy(iit, iite, std::back_inserter(dat));

    if (dat.size() <= 8 || std::string(dat.data(), 8) != "ZENCACHE") {
        log_error("zeno cache file broken (1)");
        return;
    }
    size_t pos = std::find(dat.begin() + 8, dat.end(), '\a') - dat.begin();
    if (pos == dat.size()) {
        log_error("zeno cache file broken (2)");
        return;
    }
    int keyscount = std::stoi(std::string(dat.data() + 8, pos - 8));
    pos = pos + 1;
    std::vector<std::string> keys;
    for (int k = 0; k < keyscount; k++) {
        size_t newpos = std::find(dat.begin() + pos, dat.end(), '\a') - dat.begin();
        if (newpos == dat.size()) {
            log_error("zeno cache file broken (3.{})", k);
            return;
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
            return;
        }
        const char *p = dat.data() + pos + poses[k];
        objs.try_emplace(keys[k], decodeObject(p, poses[k + 1] - poses[k]));
    }
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

    if (maxCachedFrames != 0) { // immediatedump
        int i = m_maxPlayFrame;
        toDisk(cacheFramePath, i, m_frames[i].view_objects);
        m_inCacheFrames.erase(i);
    }

    m_maxPlayFrame += 1;
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
    beginFrameNumber = 0;
    beginFrameNumber = 0;
    maxCachedFrames = 1;
    cacheFramePath = {};
}

ZENO_API void GlobalComm::frameCache(std::string const &path, int gcmax) {
    cacheFramePath = path;
    maxCachedFrames = gcmax;
}

ZENO_API void GlobalComm::frameRange(int beg, int end) {
    beginFrameNumber = beg;
    endFrameNumber = end;
}

ZENO_API int GlobalComm::maxPlayFrames() {
    std::lock_guard lck(m_mtx);
    return m_maxPlayFrame + beginFrameNumber; // m_frames.size();
}

ZENO_API GlobalComm::ViewObjects const *GlobalComm::getViewObjects(int frameid) {
    frameid -= beginFrameNumber;
    std::lock_guard lck(m_mtx);
    if (frameid < 0 || frameid >= m_frames.size())
        return nullptr;
    if (maxCachedFrames != 0) {
        // load back one gc:
        if (!m_inCacheFrames.count(frameid)) {  // notinmem then cacheit
            fromDisk(cacheFramePath, frameid, m_frames[frameid].view_objects);
            m_inCacheFrames.insert(frameid);
            // and dump one as balance:
            if (m_inCacheFrames.size() && m_inCacheFrames.size() > maxCachedFrames) { // notindisk then dumpit
                for (int i: m_inCacheFrames) {
                    if (i != frameid) {
                        toDisk(cacheFramePath, i, m_frames[i].view_objects);
                        m_inCacheFrames.erase(i);
                        break;
                    }
                }
            }
        }
    }
    return &m_frames[frameid].view_objects;
}

ZENO_API GlobalComm::ViewObjects const &GlobalComm::getViewObjects() {
    std::lock_guard lck(m_mtx);
    return m_frames.back().view_objects;
}

ZENO_API bool GlobalComm::isFrameCompleted(int frameid) const {
    if (frameid < 0 || frameid >= m_frames.size())
        return false;
    return m_frames[frameid].b_frame_completed;
}

}
