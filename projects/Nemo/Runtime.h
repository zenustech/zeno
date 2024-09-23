#pragma once

#include "Context.h"
#include <tinygltf/json.hpp>

namespace nemo {
struct ComputeTask {
  using Execute = void (*)(void *, void *);
  explicit ComputeTask(Execute execute) : execute(execute) {}
  bool dirty = true;
  Execute execute;
  std::vector<unsigned> outputs;
};

struct Runtime {
  bool singlePrecision = true;
  bool cuda = false;

  DataStorage data;
  ResourcePool resource;

  // compiled binary handle
  MOD hGetProcIDDLL = nullptr;

  std::vector<ComputeTask> tasks;

  // key: input id
  // value: tasks affected by this input
  std::vector<std::vector<unsigned>> LUT_tasks_affected;

  // key: output id
  // value: tasks affecting this input
  std::vector<std::vector<unsigned>> LUT_tasks_affecting;

public:
  ~Runtime();

  void init(const nlohmann::json &config, std::string pathConfig);

  void dirty(unsigned input);

  std::vector<unsigned> evaluate(unsigned output);

  void evaluate_all();

private:
  void load_plugs(const nlohmann::json &root);

  void load_tasks(const nlohmann::json &root);
};

std::string getAbsolutePath(std::string parent, std::string path);
} // namespace nemo
