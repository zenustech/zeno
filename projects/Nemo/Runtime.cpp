#include "Runtime.h"
#include <boost/algorithm/string.hpp>
#include "zeno/utils/format.h"
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <vector>

namespace nemo {

std::filesystem::path create_temporary_directory(unsigned long long max_tries = 1000) {
  auto tmp_dir = std::filesystem::temp_directory_path();
  unsigned long long i = 0;
  std::random_device dev;
  std::mt19937 prng(dev());
  std::uniform_int_distribution<uint64_t> rand(0);
  std::filesystem::path path;
  while (true) {
    std::stringstream ss;
    ss << std::hex << rand(prng);
    path = tmp_dir / ss.str();
    // true if the directory was created.
    if (std::filesystem::create_directory(path)) {
      break;
    }
    if (i == max_tries) {
      throw std::runtime_error("could not find non-existing directory");
    }
    i++;
  }
  return path;
}

void Runtime::init(const nlohmann::json &root, std::string pathConfig) {
  singlePrecision = root.value("singlePrecision", true);
  cuda = root.value("cuda", false);

  std::string path_bin = getAbsolutePath(pathConfig, root.at("bin"));
  bool has_ext = boost::algorithm::ends_with(path_bin, ".dll") || boost::algorithm::ends_with(path_bin, ".so");
#ifdef WIN32
  if (!has_ext)
    path_bin += ".dll";
#else
  if (!has_ext) {
    int sep = path_bin.rfind('/');
    std::string folder = path_bin.substr(0, sep + 1);
    std::string filename = path_bin.substr(sep + 1);
    if (filename.compare(0, 3, "lib")) // not start with "lib"
      filename = "lib" + filename;
    path_bin = folder + filename + ".so";
  }
#endif
  using namespace std::filesystem;
  path temp_path_bin = create_temporary_directory() / path(path_bin).filename();
  copy(path_bin, temp_path_bin);
  path_bin = temp_path_bin.string();

#ifdef WIN32
  hGetProcIDDLL = LoadLibrary(path_bin.c_str());
#else
  hGetProcIDDLL = dlopen(path_bin.c_str(), RTLD_NOW);
#endif
  if (hGetProcIDDLL == nullptr) {
    throw std::runtime_error(zeno::format("Could not load the binary: {}", path_bin));
  }
  data.init(hGetProcIDDLL);

  auto jDataBlock = root["dataBlock"];
  for (auto [key, value] : jDataBlock.items()) {
    if (!cuda && key == "Slot")
      continue;
    data.resize(key, value.get<std::size_t>(), singlePrecision);
  }

  std::string path_resource = getAbsolutePath(pathConfig, root.at("resource"));
  resource.init(hGetProcIDDLL, path_resource);

  load_plugs(root);
  load_tasks(root);
}

void Runtime::dirty(unsigned input) {
  for (unsigned id_task : LUT_tasks_affected.at(input))
    tasks.at(id_task).dirty = true;
}

std::vector<unsigned> Runtime::evaluate(unsigned id_output) {
  std::vector<unsigned> outputs;
  for (unsigned id_task : LUT_tasks_affecting.at(id_output)) {
    ComputeTask &task = tasks.at(id_task);
    if (task.dirty) {
      (task.execute)(data.instance, resource.instance);
      task.dirty = false;
      outputs.insert(outputs.end(), task.outputs.begin(), task.outputs.end());
    }
  }
  return outputs;
}

void Runtime::evaluate_all() {
  for (ComputeTask &task : tasks) {
    (task.execute)(data.instance, resource.instance);
    task.dirty = false;
  }
}

void Runtime::load_plugs(const nlohmann::json &root) {
  for (const auto &element : root["inputs"]) {
    if (element.count("tasks"))
      LUT_tasks_affected.push_back(element.at("tasks").get<std::vector<unsigned>>());
  }

  for (const auto &element : root["outputs"]) {
    if (element.count("tasks"))
      LUT_tasks_affecting.push_back(element.at("tasks").get<std::vector<unsigned>>());
  }
}

void Runtime::load_tasks(const nlohmann::json &root) {
  const auto &jTasks = root["computeTask"];
  for (auto element : jTasks) {
    unsigned id_task = tasks.size();
    ComputeTask task{get_fn<ComputeTask::Execute>(hGetProcIDDLL, zeno::format("ComputeTask{}", id_task).c_str())};
    task.outputs = element.value<std::vector<unsigned>>("outputs", {});
    tasks.push_back(task);
    if (element.value("static", false))
      (task.execute)(data.instance, resource.instance);
  }
  if (!LUT_tasks_affected.empty()) // pre-computed
    return;

  const unsigned num_inputs = root["inputs"].size();
  const unsigned num_outputs = root["outputs"].size();

  LUT_tasks_affected.resize(num_inputs);
  const unsigned num_tasks = jTasks.size();
  std::vector<std::set<unsigned>> relative_inputs_of_output(num_outputs);
  std::vector<std::set<unsigned>> inputs_of_task(num_tasks);
  for (unsigned id_task = 0; id_task != num_tasks; ++id_task) {
    const nlohmann::json &element = jTasks[id_task];
    std::vector<unsigned> task_affecings = element["inputs"];
    inputs_of_task[id_task] = {task_affecings.begin(), task_affecings.end()};
    if (element.count("outputs")) {
      for (unsigned id_data : element["outputs"]) {
        unsigned id_output = id_data - num_inputs;
        for (unsigned id_input : task_affecings)
          relative_inputs_of_output[id_output].insert(id_input);
      }
    }
    for (unsigned id_input : task_affecings) {
      LUT_tasks_affected[id_input].push_back(id_task);
    }
  }

  LUT_tasks_affecting.resize(num_outputs);
  for (unsigned id_output = 0; id_output != num_outputs; ++id_output) {
    std::set<unsigned> given_inputs = relative_inputs_of_output.at(id_output);

    // tasks depending on inputs
    std::set<unsigned> candidate_tasks;
    for (unsigned id_input : given_inputs) {
      for (unsigned id_task : LUT_tasks_affected.at(id_input)) {
        candidate_tasks.insert(id_task);
      }
    }

    std::set<unsigned> tasks;
    for (unsigned id_task : candidate_tasks) {
      std::set<unsigned> task_inputs = inputs_of_task.at(id_task);
      // only tasks satisfying "inputs no-expanding" condition can be executed
      if (std::includes(given_inputs.begin(), given_inputs.end(), task_inputs.begin(), task_inputs.end()))
        tasks.insert(id_task);
    }
    LUT_tasks_affecting[id_output] = {tasks.begin(), tasks.end()};
  }
}

std::string expandEnvironmentVariables(const std::string &s) {
  if (s.find("${") == std::string::npos)
    return s;

  std::string pre = s.substr(0, s.find("${"));
  std::string post = s.substr(s.find("${") + 2);

  if (post.find('}') == std::string::npos)
    return s;

  std::string variable = post.substr(0, post.find('}'));
  std::string value = "";

  post = post.substr(post.find('}') + 1);

  const char *v = getenv(variable.c_str());
  if (v != NULL)
    value = std::string(v);

  return expandEnvironmentVariables(pre + value + expandEnvironmentVariables(post));
}

Runtime::~Runtime() {
  if (hGetProcIDDLL == 0)
    return;

  data.cleanup();
  resource.cleanup();

  std::string error;
#ifdef WIN32
  if (0 == FreeLibrary(hGetProcIDDLL)) {
    DWORD errorMessageID = GetLastError();
    if (errorMessageID != NULL) {
      LPSTR messageBuffer = nullptr;
      size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, errorMessageID,
                                   MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);
      error = std::string(messageBuffer, size);
      LocalFree(messageBuffer);
    }
  }
#else
  if (dlclose(hGetProcIDDLL) != 0)
    error = dlerror();
#endif
  if (!error.empty())
    std::cerr << zeno::format("[Nemo]Free binary failed: {}", error) << std::endl;
}

std::string getAbsolutePath(std::string parent, std::string path) {
  std::filesystem::path p{expandEnvironmentVariables(path)};
  if (p.has_root_path())
    return p.string();
  auto base = std::filesystem::path(parent).parent_path();
  return std::filesystem::absolute(base / p).string();
}
} // namespace nemo
