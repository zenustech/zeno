#include "main.hpp"
#include "IGraphic.hpp"
#include <zeno/PrimitiveIO.h>


namespace zenvis {

std::vector<std::unique_ptr<FrameData>> frames;

std::unique_ptr<IGraphic> makeGraphicPrimitive
    ( zeno::PrimitiveObject *prim
    , std::string const &path
    );


std::unique_ptr<IGraphic> makeGraphic(std::string path, std::string ext) {
    if (ext == ".zpm") {
        auto prim = std::make_unique<zeno::PrimitiveObject>();
        zeno::readzpm(prim.get(), path.c_str());
        return makeGraphicPrimitive(prim.get(), path);

    } else {
        //printf("%s\n", ext.c_str());
        //assert(0 && "bad file extension name");
    }
    return nullptr;
}

FrameData *current_frame_data() {
    if (frames.size() < curr_frameid + 1)
        frames.resize(curr_frameid + 1);
    if (!frames[curr_frameid])
        frames[curr_frameid] = std::make_unique<FrameData>();
    return frames[curr_frameid].get();
}

void clear_graphics() {
    frames.clear();
}

void load_file(std::string name, std::string ext, std::string path, int frameid) {
    auto &graphics = current_frame_data()->graphics;
    if (graphics.find(name) != graphics.end()) {
        printf("cached: %p %s %s\n", &graphics, path.c_str(), name.c_str());
        return;
    }
    printf("load_file: %p %s %s\n", &graphics, path.c_str(), name.c_str());

    auto ig = makeGraphic(path, ext);
    if (!ig) return;
    graphics[name] = std::move(ig);
}

}
