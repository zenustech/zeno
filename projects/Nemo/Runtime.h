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
