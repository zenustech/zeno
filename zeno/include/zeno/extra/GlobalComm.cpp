#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/logger.h>

namespace zeno {

ZENO_API void GlobalComm::newFrame() {
    std::lock_guard lck(mtx);
    frames.emplace_back();
    log_debug("GlobalComm::newFrame {}", frames.size());
}

ZENO_API void GlobalComm::addViewObject(std::shared_ptr<IObject> const &object) {
    std::lock_guard lck(mtx);
    log_debug("GlobalComm::addViewObject {}", frames.size());
    frames.back().view_objects.push_back(object);
}

ZENO_API void GlobalComm::clearState() {
    std::lock_guard lck(mtx);
    frames.clear();
}

ZENO_API int GlobalComm::countFrames() {
    std::lock_guard lck(mtx);
    return frames.size();
}

ZENO_API std::vector<std::shared_ptr<IObject>> GlobalComm::getViewObjects(int frameid) {
    std::lock_guard lck(mtx);
    if (frameid < 0 || frameid >= frames.size()) return {};
    return frames[frameid].view_objects;
}

ZENO_API std::vector<std::shared_ptr<IObject>> GlobalComm::getViewObjects() {
    return frames.back().view_objects;
}

}
