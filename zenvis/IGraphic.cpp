#include "main.hpp"
#include "frames.hpp"
#include "IGraphic.hpp"
#include "IG_mesh.hpp"
//#include "IG_pars.hpp"
//#include "IG_voxl.hpp"

namespace zenvis {

std::vector<std::unique_ptr<IGraphic>> graphics;

static int last_frameid;

void update_frame_graphics() {
  if (last_frameid == curr_frameid)
    return;
  last_frameid = curr_frameid;

  graphics.clear();

  if (frames.find(curr_frameid) == frames.end()) {
    printf("frame cache invalid at frame id: %d\n", curr_frameid);
    return;
  }
  auto *frm = frames.at(curr_frameid).get();

  for (auto const &obj : frm->objects) {
    std::unique_ptr<IGraphic> gra;

    if (obj->type == "MESH") {
      gra = std::make_unique<GraphicMesh>(*obj->serial);

    //} else if (obj->type == "PARS") {
    //  gra = std::make_unique<GraphicParticles>(obj->serial);

    } else {
      printf("Bad object type: %s\n", obj->type.c_str());
      continue;
    }

    graphics.push_back(std::move(gra));
  }
}

}
