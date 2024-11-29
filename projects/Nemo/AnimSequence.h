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