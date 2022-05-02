#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/logger.h>

namespace zeno {

ZENO_API void GlobalComm::newFrame() {
    std::lock_guard lck(m_mtx);
    m_frames.emplace_back();
    log_debug("GlobalComm::newFrame {}", m_frames.size());
}

ZENO_API void GlobalComm::finishFrame() {
    std::lock_guard lck(m_mtx);
    m_maxPlayFrame += 1;
    log_debug("GlobalComm::finishFrame {}", m_maxPlayFrame);
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
    m_maxPlayFrame = 0;
}

ZENO_API int GlobalComm::maxPlayFrames() {
    std::lock_guard lck(m_mtx);
    return m_maxPlayFrame; // m_frames.size();
}

ZENO_API GlobalComm::ViewObjects const &GlobalComm::getViewObjects(int frameid) {
    std::lock_guard lck(m_mtx);
    if (frameid < 0 || frameid >= m_frames.size()) throw makeError("no frame cache at {}", frameid);
    return m_frames[frameid].view_objects;
}

ZENO_API GlobalComm::ViewObjects const &GlobalComm::getViewObjects() {
    std::lock_guard lck(m_mtx);
    return m_frames.back().view_objects;
}

}
