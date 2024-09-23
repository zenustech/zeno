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

  // value: plug id
  std::vector<unsigned> meshes;

  // key: output id of mesh
  // value: full path
  std::map<unsigned, std::string> LUT_path;

  // key: output id
  // value: topology
  std::map<unsigned, unsigned> LUT_topology;

  // key: output id
  // value: uvs
  std::map<unsigned, std::vector<std::pair<std::string, unsigned>>> LUT_uvsets;

  // key: mesh out plug id
  // value: dataLocation(dynamic) or worldMatrix(static)
  std::map<unsigned, std::variant<unsigned, glm::mat4>> LUT_transform;

public:
  Evaluator(std::string path_config, std::string path_anim);

  void evaluate(float frame);

  std::pair<int, int> duration() const { return animation.duration; }

  std::pair<std::vector<unsigned>, std::vector<unsigned>> getTopo(unsigned plug_id) { return runtime.resource.getTopo(LUT_topology.at(plug_id)); }

  std::tuple<std::vector<float>, std::vector<float>, std::vector<unsigned>> getDefaultUV(unsigned plug_id) {
    return runtime.resource.getUV(LUT_uvsets.at(plug_id).front().second);
  }

  std::vector<glm::vec3> getPoints(unsigned plug_id) const;

private:
  void load_plugs(const nlohmann::json &root);

  void load_topology(const nlohmann::json &root);

  void update_inputs(float frame);
};
} // namespace nemo