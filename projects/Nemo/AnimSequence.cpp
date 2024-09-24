/*
 * MIT License
 *
 * Copyright (c) 2024 wuzhen
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * 1. The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 * 2. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *    SOFTWARE.
 */

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
