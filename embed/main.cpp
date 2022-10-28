#include <zeno/core/Graph.h>
#include <zeno/core/Session.h>

int main() {
    const char *json = "json here";
    auto g = zeno::getSession().createGraph();
    g->addSubnetNode("CustomMain")->loadGraph(json);
    std::map<std::string, std::shared_ptr<zeno::IObject>> inputs, outputs;
    outputs = g->callTempNode("CustomMain", std::move(inputs));
    return 0;
}
