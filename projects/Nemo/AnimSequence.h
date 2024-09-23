#pragma once
#include <glm/glm.hpp>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace nemo {
using ChannelValue = std::variant<glm::dvec3, glm::dmat4, double>;
struct Channel {
  std::string name;
  std::map<int, ChannelValue> frames;
};

struct Animation {
  std::vector<Channel> channels;
  std::pair<int, int> duration;

  void load(std::string path);

  template <typename T> T get(unsigned id, float time) {
    int framebegin = channels[id].frames.begin()->first;
    int frameend = channels[id].frames.rbegin()->first;
    time = std::max<float>(time, framebegin);
    time = std::min<float>(time, frameend);

    if (isIntegral(time))
      return std::get<T>(channels[id].frames.at(static_cast<int>(time)));

    float beg = std::floor(time);
    float end = std::ceil(time);
    double alpha = (time - beg) / (end - beg);
    return (1.0 - alpha) * std::get<T>(channels[id].frames.at(static_cast<int>(beg))) + alpha * std::get<T>(channels[id].frames.at(static_cast<int>(end)));
  }

private:
  static bool isIntegral(float x) { return std::abs(x - std::roundf(x)) < 1E-5F; }
};
} // namespace nemo