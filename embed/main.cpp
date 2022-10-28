#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>

int main() {
    const char *json = "json here";
    auto g = zeno::getSession().createGraph();
    g->addSubnetNode("custom")->loadGraph(json);
    std::map<std::string, std::shared_ptr<zeno::IObject>> inputs, outputs;
    outputs = g->callSubnetNode("custom", std::move(inputs));
    return 0;
}
