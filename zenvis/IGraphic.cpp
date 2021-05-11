#include "main.hpp"
#include "IGraphic.hpp"
#include <zen/PrimitiveIO.h>


namespace zenvis {

std::unique_ptr<IGraphic> makeGraphicMesh(ObjectData const &obj);
std::unique_ptr<IGraphic> makeGraphicParticles(ObjectData const &obj);

std::vector<std::unique_ptr<IGraphic>> graphics;


void load_file(std::string name, std::string ext, std::string path, int frameid) {
    printf("load_file: %s\n", path.c_str());

    if (ext == ".zpm") {
        auto prim = std::make_unique<zenbase::PrimitiveObject>();
        zenbase::readzpm(prim.get(), path);
        auto &pos = prim->attr<zen::vec3f>("pos");

    } else {
        printf("%s\n", ext.c_str());
        assert(0 && "bad file extension name");
    }
}

void update_frame_graphics() {
    static int last_frameid = -2;
    if (last_frameid == curr_frameid)
        return;
    last_frameid = curr_frameid;

    graphics.clear();

    if (frames.find(curr_frameid) == frames.end()) {
        printf("no frame cache at frame id: %d\n", curr_frameid);
        return;
    }
    auto *frm = frames.at(curr_frameid).get();

    for (auto const &obj : frm->objects) {
        std::unique_ptr<IGraphic> gra;

        if (obj->type == "MESH") {
            gra = makeGraphicMesh(*obj);

        } else if (obj->type == "PARS") {
            gra = makeGraphicParticles(*obj);

        } else {
            printf("Bad object type: %s\n", obj->type.c_str());
            continue;
        }

        graphics.push_back(std::move(gra));
    }
}

}
