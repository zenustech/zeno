#include "AnimSequence.h"
#include <fstream>
#include <tinygltf/json.hpp>

namespace nemo {
void Animation::load(std::string path) {
  duration.first = INT_MAX;
  duration.second = INT_MIN;
  channels.clear();

  std::ifstream fin(path);
  if (!fin.is_open())
    throw std::runtime_error("Could not load animation: " + path);
  nlohmann::json root = nlohmann::json::parse(fin);
  for (const auto &element : root.items()) {
    Channel channel;
    channel.name = element.key();
    std::string type = element.value()["type"];
    for (const auto &frame : element.value()["values"].items()) {
      int key = std::stoi(frame.key());
      duration.first = std::min(duration.first, key);
      duration.second = std::max(duration.second, key);
      ChannelValue value;
      nlohmann::json j = frame.value();
      if (type == "matrix") {
        glm::dmat4 matrix;
        for (int i = 0; i != 16; ++i)
          matrix[i / 4][i % 4] = j[i].get<double>();
        value = matrix;
      } else if (type == "double3") {
        value = glm::dvec3{j[0].get<double>(), j[1].get<double>(), j[2].get<double>()};
      } else if (type == "double") {
        if (j.is_boolean())
          value = static_cast<double>(j.get<bool>());
        else
          value = j.get<double>();
      } else {
        throw std::runtime_error("unknown type: " + type);
      }
      channel.frames.insert(std::make_pair(key, value));
    }
    channels.push_back(std::move(channel));
  }
}
} // namespace nemo
