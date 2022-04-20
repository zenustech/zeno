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

ZENO_API void GlobalComm::addViewObject(std::shared_ptr<IObject> const &object) {
    std::lock_guard lck(m_mtx);
    log_debug("GlobalComm::addViewObject {}", m_frames.size());
    if (m_frames.empty()) throw makeError("empty frame cache");
    m_frames.back().view_objects.push_back(object);
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

ZENO_API std::vector<std::shared_ptr<IObject>> GlobalComm::getViewObjects(int frameid) {
    std::lock_guard lck(m_mtx);
    if (frameid < 0 || frameid >= m_frames.size()) return {};
    return m_frames[frameid].view_objects;
}

ZENO_API std::vector<std::shared_ptr<IObject>> GlobalComm::getViewObjects() {
    std::lock_guard lck(m_mtx);
    return m_frames.back().view_objects;
}

}
