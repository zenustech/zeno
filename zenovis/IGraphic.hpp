#pragma once

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <zeno/core/IObject.h>

namespace zenvis {

struct IGraphic {
  virtual void draw() = 0;
  virtual ~IGraphic() = default;
};

struct FrameData {
    std::map<std::shared_ptr<zeno::IObject>, std::unique_ptr<IGraphic>> graphics;
};

FrameData *current_frame_data();

extern std::vector<std::unique_ptr<FrameData>> frames;

}
