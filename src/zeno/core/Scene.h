#pragma once

#include <zeno/utils/api.h>
#include <memory>
#include <string>
#include <map>

namespace zeno {

struct Session;
struct Graph;

struct Scene {
    std::map<std::string, std::unique_ptr<Graph>> graphs;

    Graph *currGraph = nullptr;
    Session *sess = nullptr;

    ZENO_API Scene();
    ZENO_API ~Scene();

    Scene(Scene const &) = delete;
    Scene &operator=(Scene const &) = delete;
    Scene(Scene &&) = delete;
    Scene &operator=(Scene &&) = delete;

    ZENO_API Graph &getGraph();
    ZENO_API void clearAllState();
    ZENO_API Graph &getGraph(std::string const &name) const;
    ZENO_API void switchGraph(std::string const &name);
    ZENO_API void loadScene(const char *json);
};

}
