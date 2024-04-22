#include "Task.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <queue>
#include <stack>
#include <tinygltf/json.hpp>

namespace zeno {

struct WriteTaskDependencyGraph : INode {
  using Json = nlohmann::json;
  static void process_node(std::vector<Json> &jsons, WorkNode *node,
                           std::set<WorkNode *> &records) {
    if (records.find(node) != records.end())
      return;
    records.insert(node);

    for (auto &&[tag, node] : node->deps)
      process_node(jsons, node.get(), records);

    if (!node->tag.empty()) { // skip empty job
      Json json;
      json["name"] = node->tag;
      json["cmds"] = node->workItems;
      std::vector<std::string> depWorkNames;
      for (auto &&[tag, node] : node->deps)
        depWorkNames.push_back(node->tag);
      json["deps"] = depWorkNames;
      Json j;
      j[node->tag] = json;
      jsons.push_back(j);
    }
  }
  static void process_node_bfs(std::vector<Json> &jsons, WorkNode *node,
                               std::set<WorkNode *> &records) {
    std::queue<WorkNode *> que;
    std::stack<WorkNode *> orderedNodes;

    que.push(node);

    while (!que.empty()) {
      auto n = que.front();
      que.pop();

      if (records.find(n) != records.end())
        continue;
      records.insert(n);

      orderedNodes.push(n);

      for (auto &&[tag, no] : n->deps)
        que.push(no.get());
    }

    while (!orderedNodes.empty()) {
      auto n = orderedNodes.top();
      orderedNodes.pop();

      if (!n->tag.empty()) { // skip empty job
        Json json;
        json["name"] = n->tag;
        json["cmds"] = n->workItems;
        std::vector<std::string> depWorkNames;
        for (auto &&[tag, node] : n->deps)
          depWorkNames.push_back(node->tag);
        json["deps"] = depWorkNames;
        Json j;
        j[n->tag] = json;
        jsons.push_back(j);
      }
    }
  }

  static void process_node_topo(std::vector<Json> &jsons, WorkNode *node,
                                std::set<WorkNode *> &records) {
    puts("Iteration");
    std::map<WorkNode *, int> indegrees;
    std::map<WorkNode *, std::vector<WorkNode *>> successors;
    std::map<WorkNode *, int> depths;
    /// calculate depths and initial indegrees
    std::set<WorkNode *> visited;
    std::queue<WorkNode *> que;

    que.push(node);
    depths[node] = 0;

    while (!que.empty()) {
      auto n = que.front();
      que.pop();

      if (visited.find(n) != visited.end())
        continue;
      visited.insert(n);

      int indD = n->deps.size(), numVisited = 0;

      auto d = depths[n];
      for (auto &&[tag, no] : n->deps) {
        if (records.find(no.get()) != records.end()) {
          numVisited++;
        } else {
          que.push(no.get());
          //
          successors[no.get()].push_back(n);
          depths[no.get()] = d + 1;
        }
      }
      indegrees[n] = indD - numVisited;
    }

    /// topology sort
    std::vector<WorkNode *> zeroInDegree;
    zeroInDegree.reserve(visited.size());
    std::vector<WorkNode *> expandCandidates;
    expandCandidates.reserve(visited.size()); // double buffer
    std::vector<WorkNode *> orderedNodes;     // result

    // gather candidates
    for (const auto &[n, inDegree] : indegrees)
      if (inDegree == 0 && records.find(n) == records.end()) {
        zeroInDegree.push_back(n);
        records.insert(n);
      }

    while (!zeroInDegree.empty()) {
#if 0
      fmt::print("\tnext batch\n\t");
      for (const auto &n : zeroInDegree) {
        fmt::print("{} ", n->tag);
      }
      fmt::print("\n");
#endif

      // update indegrees
      for (const auto &n : zeroInDegree) {
        for (const auto &suc : successors[n])
          --indegrees[suc];
      }
      // further order candidates by their depths
      std::sort(std::begin(zeroInDegree), std::end(zeroInDegree),
                [&depths](WorkNode *l, WorkNode *r) {
                  return depths[l] > depths[r];
                });
#if 0
      fmt::print("\t\tordered batch\n\t\t");
      for (const auto &n : zeroInDegree) {
        fmt::print("{} ", n->tag);
      }
      fmt::print("\n");
#endif
      // gather candidates
      expandCandidates.clear();
      for (const auto &n : zeroInDegree) {
        orderedNodes.push_back(n); // result

        // fmt::print("expanding {}: ", n->tag);
        for (const auto &suc : successors[n]) {
          // fmt::print("->[{}] ", suc->tag);
          if (indegrees[suc] == 0 && records.find(suc) == records.end()) {
            expandCandidates.push_back(suc);
            records.insert(suc);
            // fmt::print("accept; ");
          } //  else
            // fmt::print("reject (ind {}, inserted {}); ", indegrees[suc],
            //            records.find(suc) != records.end());
        }
        // fmt::print("\n");
      }
      std::swap(zeroInDegree, expandCandidates);
    }

    // if (orderedNodes.size() != nNodes)
    //   fmt::print("there exists loop in the dependency graph.");

    for (auto n : orderedNodes) {
      if (!n->tag.empty()) { // skip empty job
        Json json;
        json["name"] = n->tag;
        json["cmds"] = n->workItems;
        std::vector<std::string> depWorkNames;
        for (auto &&[tag, node] : n->deps)
          depWorkNames.push_back(node->tag);
        json["deps"] = depWorkNames;
        Json j;
        j[n->tag] = json;
        jsons.push_back(j);
      }
    }
  }
  void apply() override {
    std::vector<WorkNode *> nodes;
    auto jobs = get_input("job");
    if (auto ptr = std::dynamic_pointer_cast<WorkNode>(jobs); ptr)
      nodes.push_back(ptr.get());
    else if (auto list = std::dynamic_pointer_cast<ListObject>(jobs); list) {
      for (auto &&arg : list->get())
        if (auto ptr = std::dynamic_pointer_cast<WorkNode>(arg); ptr)
          nodes.push_back(ptr.get());
    }
    auto filename = get_input2<std::string>("json_file_path");

    std::vector<Json> jsons;

    std::set<WorkNode *> records;
    for (auto &node : nodes)
      process_node_topo(jsons, node, records);

    Json json = Json(jsons);

    std::ofstream file(filename);
    if (file.is_open()) {
      // dump(4) prints the JSON data with an indentation of 4 spaces
      file << json.dump(4);
      file.close();
      // fmt::print("Task Dependency Graph [{}] written to {} in json\n",
      //           node->tag, filename);
    } else {
      throw std::runtime_error(
          fmt::format("Could not open file [{}] for writing.", filename));
    }
    set_output("job", jobs);
  }
};
ZENO_DEFNODE(WriteTaskDependencyGraph)
({/* inputs: */
  {
      {"list", "job"},
      {"writepath", "json_file_path", ""},
  },
  /* outputs: */
  {
      {"job"},
  },
  /* params: */
  {},
  /* category: */
  {
      "task",
  }});

struct AssembleJob : INode {
  void apply() override {
    auto ret = std::make_shared<WorkNode>();

    auto tag = get_input2<std::string>("name_tag");
    if (tag.empty())
      throw std::runtime_error("work name must not be empty!");
    ret->tag = tag;

    std::vector<std::string> workItems;
    auto cmds = get_input("scripts");
    if (auto ptr = std::dynamic_pointer_cast<StringObject>(cmds); ptr)
      workItems.push_back(ptr->get());
    else if (auto list = std::dynamic_pointer_cast<ListObject>(cmds); list) {
      for (auto &&arg : list->get())
        if (auto ptr = std::dynamic_pointer_cast<StringObject>(arg); ptr)
          workItems.push_back(ptr->get());
    }
    ret->workItems = workItems;

    auto deps = has_input("dependencies")
                    ? get_input<ListObject>("dependencies")
                    : std::make_shared<ListObject>();
    for (auto &&arg : deps->get())
      if (auto ptr = std::dynamic_pointer_cast<WorkNode>(arg); ptr)
        ret->deps[ptr->tag] = ptr;

    set_output("job", ret);
  }
};
ZENO_DEFNODE(AssembleJob)
({/* inputs: */
  {
      {"string", "name_tag"},
      {"list", "scripts"},
      {"list", "dependencies"},
  },
  /* outputs: */
  {
      {"WorkNode", "job"},
  },
  /* params: */
  {},
  /* category: */
  {
      "task",
  }});

struct SetWorkDependencies : INode {
  void apply() override {
    auto node = get_input<WorkNode>("job");
    auto reset = get_input2<bool>("reset");
    if (reset)
      node->deps.clear();

    auto deps = has_input("dependencies")
                    ? get_input<ListObject>("dependencies")
                    : std::make_shared<ListObject>();
    for (auto &&arg : deps->get())
      if (auto ptr = std::dynamic_pointer_cast<WorkNode>(arg); ptr)
        node->deps[ptr->tag] = ptr;

    set_output("job", node);
  }
};
ZENO_DEFNODE(SetWorkDependencies)
({/* inputs: */
  {
      {"WorkNode", "job"},
      {"list", "dependencies"},
      {"bool", "reset", "false"},
  },
  /* outputs: */
  {
      {"WorkNode", "job"},
  },
  /* params: */
  {},
  /* category: */
  {
      "task",
  }});

} // namespace zeno