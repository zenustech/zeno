#pragma once

#include <memory>
#include <vector>

namespace zenvis {

struct IGraphic {
  virtual void draw() = 0;
};

extern std::vector<std::unique_ptr<IGraphic>> graphics;

void update_frame_graphics();

}
