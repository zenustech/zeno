#pragma once

#include <Light.hpp>
#include <memory>
#include <vector>
#include <string>
#include <map>

namespace zenvis {

struct IGraphic {
  float m_weight;
  void setMultiSampleWeight(float w)
  {
    m_weight = w;
  }
  virtual void draw(bool reflect, float depthPass) = 0;
  virtual void drawShadow(Light *light) = 0;
  virtual void drawVoxelize(){};
  virtual ~IGraphic() = default;
};

struct FrameData {
    std::map<std::string, std::unique_ptr<IGraphic>> graphics;
};

FrameData *current_frame_data();

/* extern std::vector<std::unique_ptr<FrameData>> frames; */

}
