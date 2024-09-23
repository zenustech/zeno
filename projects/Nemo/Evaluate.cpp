#include "Evaluate.h"
#include "zeno/utils/log.h"
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <boost/algorithm/string.hpp>

namespace glm {
inline void from_json(const nlohmann::json &j, mat4 &mat) {
  for (int i = 0; i != 16; ++i)
    mat[i / 4][i % 4] = j[i].get<float>();
}
} // namespace glm

namespace nemo {
Evaluator::Evaluator(std::string path_config, std::string path_anim) {
  // load config
  {
    std::ifstream fin(path_config);
    if (!fin.is_open())
      throw std::runtime_error("Could not load config: " + path_config);
    nlohmann::json config = nlohmann::json::parse(fin);
    runtime.init(config, path_config);
    load_plugs(config);
    load_topology(config);
  }

  // load anim
  {
    animation.load(path_anim);
    for (const Channel &channel : animation.channels) {
      auto iter = std::find_if(plugs.begin(), plugs.end(), [&channel](const PlugInfo &info) { return info.name == channel.name; });
      inputs.push_back(std::distance(plugs.begin(), iter));
    }
  }
}

void Evaluator::evaluate(float frame) {
  update_inputs(frame);
  runtime.evaluate_all();
}

void Evaluator::load_plugs(const nlohmann::json &root) {
  std::vector<nlohmann::json> all_ios = root["inputs"];
  for (auto x : root["outputs"])
    all_ios.push_back(x);

  for (const auto &element : all_ios) {
    PlugInfo info;
    info.name = element.at("name");
    info.dataIndex = element.at("data_id");
    info.dataTypeStr = element.at("type");
    plugs.push_back(info);
    if (info.dataTypeStr == "Mesh" || info.dataTypeStr == "CuShape") {
      meshes.push_back(plugs.size() - 1);
    }
  }
}

void Evaluator::load_topology(const nlohmann::json &root) {
  for (auto element : root.at("topology")) {
    unsigned plug_id = element.at("vtx");
    LUT_path[plug_id] = element.value("path", "");
    LUT_topology[plug_id] = element.at("topo").get<unsigned>();
    if (element.count("uv")) {
      LUT_uvsets[plug_id].push_back(std::make_pair("map1", element.at("uv").get<unsigned>()));
    } else {
      for (const auto &jItem : element.at("uvs")) {
        LUT_uvsets[plug_id].push_back(std::make_pair(jItem.at("name").get<std::string>(), jItem.at("id").get<unsigned>()));
      }
    }

    auto jTransform = element.at("transform");
    if (jTransform.is_null())
      LUT_transform[plug_id] = glm::identity<glm::mat4>();
    else if (jTransform.is_array())
      LUT_transform[plug_id] = jTransform.get<glm::mat4>();
    else
      LUT_transform[plug_id] = jTransform.get<unsigned>();
  }
}

void Evaluator::update_inputs(float frame) {
  for (unsigned channel_id = 0; channel_id != inputs.size(); ++channel_id) {
    unsigned plug_id = inputs[channel_id];
    const PlugInfo &info = plugs[plug_id];
    if ("Bool" == info.dataTypeStr) {
      runtime.data.setBool(info.dataIndex, animation.get<double>(channel_id, frame));
    } else if ("Decimal" == info.dataTypeStr) {
      if (runtime.singlePrecision)
        runtime.data.setFloat(info.dataIndex, animation.get<double>(channel_id, frame));
      else
        runtime.data.setDouble(info.dataIndex, animation.get<double>(channel_id, frame));
    } else if ("Angle" == info.dataTypeStr) {
      if (runtime.singlePrecision)
        runtime.data.setFloat(info.dataIndex, glm::radians(animation.get<double>(channel_id, frame)));
      else
        runtime.data.setDouble(info.dataIndex, glm::radians(animation.get<double>(channel_id, frame)));
    } else if ("Int" == info.dataTypeStr) {
      runtime.data.setInt(info.dataIndex, animation.get<double>(channel_id, frame));
    } else if ("Vec3" == info.dataTypeStr) {
      if (runtime.singlePrecision)
        runtime.data.setVec3(info.dataIndex, animation.get<glm::dvec3>(channel_id, frame));
      else
        runtime.data.setDVec3(info.dataIndex, animation.get<glm::dvec3>(channel_id, frame));
    } else if ("Euler" == info.dataTypeStr) {
      if (runtime.singlePrecision)
        runtime.data.setVec3(info.dataIndex, glm::radians(animation.get<glm::dvec3>(channel_id, frame)));
      else
        runtime.data.setDVec3(info.dataIndex, glm::radians(animation.get<glm::dvec3>(channel_id, frame)));
    } else if ("Mat4" == info.dataTypeStr) {
      if (runtime.singlePrecision)
        runtime.data.setMat4(info.dataIndex, animation.get<glm::dmat4>(channel_id, frame));
      else
        runtime.data.setDMat4(info.dataIndex, animation.get<glm::dmat4>(channel_id, frame));
    } else {
      throw std::runtime_error("unknown input type: " + info.dataTypeStr);
    }
  }
}

std::vector<glm::vec3> Evaluator::getPoints(unsigned plug_id) const {
  std::vector<glm::vec3> points;
  unsigned data_id = plugs[plug_id].dataIndex;
  std::string dataType = plugs[plug_id].dataTypeStr;
  if (runtime.singlePrecision) {
    if (dataType == "CuShape")
      runtime.data.pullCuShape(data_id, points);
    else
      runtime.data.getMesh(data_id, points);
  } else {
    if (dataType == "CuShape") {
      runtime.data.pullDCuShape(data_id, points);
    } else {
      std::vector<glm::dvec3> _points;
      runtime.data.getDMesh(data_id, _points);
      points.resize(_points.size());
      for (int i = 0; i != points.size(); ++i)
        points[i] = _points[i];
    }
  }

  glm::mat4 transform;
  {
    auto item = LUT_transform.at(plug_id);
    if (item.index() == 0) {
      unsigned data_id = std::get<unsigned>(item);
      if (runtime.singlePrecision)
        transform = runtime.data.getMat4(data_id);
      else
        transform = runtime.data.getDMat4(data_id);
    } else {
      transform = std::get<glm::mat4>(item);
    }
  }

  glm::mat3 basis = transform;
  glm::vec3 offset = transform[3];
  for (auto &pnt : points) {
    pnt = basis * pnt + offset;
  }

  return points;
}
} // namespace nemo
