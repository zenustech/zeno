#include "main.hpp"
#include "IGraphic.hpp"
//#include <zeno/types/PrimitiveIO.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/cppdemangle.h>
#include <unordered_map>
#include <unordered_set>


namespace zenvis {

using namespace zeno;

static std::unordered_map<std::shared_ptr<IObject>, std::unique_ptr<IGraphic>> graphics;

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

/*FrameData *current_frame_data() {
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
            int endi = std::min(curr_frameid + nkeep / 2, (int)frames.size());
            int begi = std::max(endi - nkeep, 0);
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
}*/

std::vector<IGraphic *> current_graphics() {
    std::vector<IGraphic *> ret;
    ret.reserve(graphics.size());
    for (auto const &[obj, gra]: graphics) {
        log_trace("current_graphics: got object {} graphics {}", obj.get(), gra.get());
        ret.push_back(gra.get());
    }
    return ret;
}

void clear_graphics() {
    log_trace("clear all graphics");
    graphics.clear();
}

void load_objects(std::vector<std::shared_ptr<zeno::IObject>> const &objs) {
    static std::unordered_map<std::shared_ptr<IObject>, std::unique_ptr<IGraphic>> new_graphics;
    new_graphics.clear();
    for (auto const &obj: objs) {
        log_trace("load_object: got object {}", obj.get());
        auto it = graphics.find(obj);
        if (it != graphics.end()) {
            log_trace("load_object: cache hit graphics {}", it->second.get());
            new_graphics.emplace(it->first, std::move(it->second));
            graphics.erase(it);
            continue;
        }
        auto ig = makeGraphic(obj);
        log_trace("load_object: fresh load graphics {}", ig.get());
        if (!ig) continue;
        new_graphics.emplace(obj, std::move(ig));
    }
    std::swap(new_graphics, graphics);
    new_graphics.clear();
}

}
