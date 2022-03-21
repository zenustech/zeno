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

std::vector<IGraphic *> current_graphics();

}
