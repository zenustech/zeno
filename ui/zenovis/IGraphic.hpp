#pragma once

#include <Light.hpp>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <zeno/core/IObject.h>

namespace zenvis {

struct IGraphic {
  float m_weight;
  void setMultiSampleWeight(float w)
  {
    m_weight = w;
  }
  virtual void draw(bool reflect, float depthPass) = 0;
  virtual void drawShadow(Light *light) = 0;
  virtual ~IGraphic() = default;
};

std::vector<IGraphic *> current_graphics();

}
