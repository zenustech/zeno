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

#include "AnimSequence.h"
#include "Runtime.h"

namespace nemo {
struct Evaluator {
  Runtime runtime;
  Animation animation;

  struct PlugInfo {
    std::string name;
    unsigned dataIndex;
    std::string dataTypeStr;
  };
  std::vector<PlugInfo> plugs;

  // value: input id
  std::vector<unsigned> inputs;

  // value: mesh id
  std::vector<unsigned> meshes;

  // key: mesh id
  // value: full path
  std::map<unsigned, std::string> LUT_path;

  // key: mesh id
  // value: topology
  std::map<unsigned, unsigned> LUT_topology;

  // key: mesh id
  // value: uvs
  std::map<unsigned, std::vector<std::pair<std::string, unsigned>>> LUT_uvsets;

  // key: mesh id
  // value: (dataLocation(-1 if constant), type)
  std::map<unsigned, std::pair<int, std::string>> LUT_visible;

  // key: mesh id
  // value: dataLocation(dynamic) or worldMatrix(static)
  std::map<unsigned, std::variant<unsigned, glm::mat4>> LUT_transform;

  struct FaceSet {
    std::string name;
    // key: mesh_id
    // value: faces(full if empty)
    std::map<unsigned, std::vector<unsigned>> members;
  };
  std::vector<FaceSet> facesets;

public:
  Evaluator(std::string path_config, std::string path_anim);

  void evaluate(float frame);

  std::pair<int, int> duration() const { return animation.duration; }

  std::pair<std::vector<unsigned>, std::vector<unsigned>> getTopo(unsigned plug_id) { return runtime.resource.getTopo(LUT_topology.at(plug_id)); }

  std::tuple<std::vector<float>, std::vector<float>, std::vector<unsigned>> getDefaultUV(unsigned plug_id) {
    return runtime.resource.getUV(LUT_uvsets.at(plug_id).front().second);
  }

  bool isVisible(unsigned plug_id) const;

  std::vector<glm::vec3> getPoints(unsigned plug_id) const;

private:
  void load_plugs(const nlohmann::json &root);

  void load_topology(const nlohmann::json &root);

  void load_faceset(const nlohmann::json &root);

  void update_inputs(float frame);
};
} // namespace nemo