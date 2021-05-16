#include "main.hpp"
#include "IGraphic.hpp"
#include <zen/PrimitiveIO.h>


namespace zenvis {

std::vector<std::unique_ptr<IGraphic>> graphics;

std::unique_ptr<IGraphic> makeGraphicPrimitive
    ( std::vector<zen::vec3f> const &pos
    , std::vector<zen::vec3f> const &clr
    , std::string const &path
    );


std::unique_ptr<IGraphic> makeGraphic(std::string path, std::string ext) {
    if (ext == ".zpm") {
        auto prim = std::make_unique<zenbase::PrimitiveObject>();
        zenbase::readzpm(prim.get(), path.c_str());
        auto &pos = prim->add_attr<zen::vec3f>("pos");
        auto &clr = prim->add_attr<zen::vec3f>("clr");
        return makeGraphicPrimitive(pos, clr, path);

    } else {
        //printf("%s\n", ext.c_str());
        //assert(0 && "bad file extension name");
    }
    return nullptr;
}


void clear_graphics() {
    graphics.clear();
}

void load_file(std::string name, std::string ext, std::string path, int frameid) {
    //printf("load_file: %s\n", path.c_str());

    auto ig = makeGraphic(path, ext);
    if (ig != nullptr)
      graphics.push_back(std::move(ig));
}

}
