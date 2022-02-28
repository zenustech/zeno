#include "main.hpp"
#include "IGraphic.hpp"
//#include <zeno/types/PrimitiveIO.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/cppdemangle.h>


namespace zenvis {

using namespace zeno;

std::vector<std::unique_ptr<FrameData>> frames;

std::unique_ptr<IGraphic> makeGraphicPrimitive(std::shared_ptr<IObject> obj);
#ifdef ZENVIS_WITH_OPENVDB
std::unique_ptr<IGraphic> makeGraphicVolume(std::shared_ptr<IObject> obj);
#endif


std::unique_ptr<IGraphic> makeGraphic(std::shared_ptr<IObject> obj) {
    if (auto ig = makeGraphicPrimitive(obj)) {
        log_trace("load_object: primitive");
        return ig;
    }
#ifdef ZENVIS_WITH_OPENVDB
    if (auto ig = makeGraphicVolume(obj)) {
        log_trace("load_object: volume");
        return ig;
    }
#endif

    log_debug("load_object: unexpected view object {}", cppdemangle(typeid(*obj)));

    //printf("%s\n", ext.c_str());
    //assert(0 && "bad file extension name");
    return nullptr;
}

FrameData *current_frame_data() {
    if (frames.size() < curr_frameid + 1)
        frames.resize(curr_frameid + 1);
    if (!frames[curr_frameid])
        frames[curr_frameid] = std::make_unique<FrameData>();
    return frames[curr_frameid].get();
}

void auto_gc_frame_data(int nkeep) {
    for (int i = 0; i < frames.size(); i++) {
        auto const &frame = frames[i];
        if (frame) {
            auto endi = std::min(curr_frameid + nkeep / 2, (int)frames.size());
            auto begi = std::max(endi - nkeep, 0);
            if (i >= endi || i < begi) {
                //printf("auto gc free %d\n", i);
                frames[i] = nullptr;
            }
        }
    }
}

std::vector<int> get_valid_frames_list() {
    std::vector<int> res;
    for (int i = 0; i < frames.size(); i++) {
        if (frames[i])
            res.push_back(i);
    }
    return res;
}

void clear_graphics() {
    frames.clear();
}

void load_object(std::shared_ptr<IObject> obj, int unused_frameid) {
    auto &graphics = current_frame_data()->graphics;
    if (graphics.find(obj) != graphics.end()) {
        log_trace("load_object: using cached");
        //printf("cached: %p %s %s\n", &graphics, path.c_str(), name.c_str());
        return;
    }
    //printf("load_file: %p %s %s\n", &graphics, path.c_str(), name.c_str());

    auto ig = makeGraphic(obj);
    if (!ig) return;
    graphics.emplace(obj, std::move(ig));
}

}
