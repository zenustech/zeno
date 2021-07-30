#include <zeno/core/Scene.h>
#include <zeno/core/Graph.h>
#include <zeno/utils/safe_at.h>

namespace zeno {

ZENO_API Scene::Scene() = default;
ZENO_API Scene::~Scene() = default;

ZENO_API void Scene::clearAllState() {
    graphs.clear();
}

ZENO_API void Scene::switchGraph(std::string const &name) {
    if (graphs.find(name) == graphs.end()) {
        auto subg = std::make_unique<zeno::Graph>();
        subg->scene = this;
        graphs[name] = std::move(subg);
    }
    currGraph = graphs.at(name).get();
}

ZENO_API Graph &Scene::getGraph() {
    if (!currGraph)
        switchGraph("main");
    return *currGraph;
}

ZENO_API Graph &Scene::getGraph(std::string const &name) const {
    return *safe_at(graphs, name, "graph");
}

}
