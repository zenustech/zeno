#include <functional>
#include <typeinfo>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <set>
#include <any>




struct Context {
    std::vector<std::any> inputs;
    std::vector<std::any> outputs;
};


struct Session {
    std::map<std::string, std::function<void(Context &)>> nodes;
    std::map<int, std::any> objects;
} session;



struct Invocation {
    std::string node_name;
    std::vector<int> inputs;
    std::vector<int> outputs;

    void operator()() {
        auto const &node = session.nodes.at(node_name);
        Context ctx;
        ctx.inputs.resize(inputs.size());
        for (int i = 0; i < inputs.size(); i++) {
            ctx.inputs[i] = session.objects.at(inputs[i]);
        }
        ctx.outputs.resize(outputs.size());
        node(ctx);
        for (int i = 0; i < outputs.size(); i++) {
            session.objects[outputs[i]] = ctx.outputs[i];
        }
    }
};


void myadd(Context &ctx) {
    auto x = std::any_cast<int>(ctx.inputs[0]);
    auto y = std::any_cast<int>(ctx.inputs[1]);
    auto z = x + y;
    ctx.outputs[0] = z;
}


struct Graph {
    struct Node {
        std::string name;
        std::vector<std::pair<int, int>> inputs;
    };
    std::vector<Node> nodes;
};


struct ReverseSorter {
    std::set<int> visited;
    std::map<int, std::vector<int>> rev_links;

    std::vector<int> result;

    void build(Graph const &graph) {
        for (int dst_node = 0; dst_node < graph.nodes.size(); dst_node++) {
            auto const &node = graph.nodes.at(dst_node);
            for (auto const &[src_node, src_sock]: node.inputs) {
                rev_links[src_node].push_back(dst_node);
            }
        }
    }

    void touch(int key) {
        if (auto it = visited.find(key); it != visited.end()) {
            return;
        }
        result.push_back(key);
        visited.insert(key);
        if (auto it = rev_links.find(key); it != rev_links.end()) {
            for (auto const &target: it->second) {
                touch(target);
            }
        }
    }
};


struct ForwardSorter {
    std::set<int> visited;
    std::map<int, std::vector<int>> links;

    std::vector<int> result;

    void build(Graph const &graph) {
        for (int dst_node = 0; dst_node < graph.nodes.size(); dst_node++) {
            auto &link = links[dst_node];
            auto const &node = graph.nodes.at(dst_node);
            for (auto const &[src_node, src_sock]: node.inputs) {
                link.push_back(src_node);
            }
        }
    }

    void touch(int key) {
        if (auto it = visited.find(key); it != visited.end()) {
            return;
        }
        visited.insert(key);
        if (auto it = links.find(key); it != links.end()) {
            for (auto const &source: it->second) {
                touch(source);
            }
        }
        result.push_back(key);
    }
};


int main() {
    Graph graph;
    graph.nodes.push_back({"myfunc", {}});
    graph.nodes.push_back({"hisfunc", {{0, 0}}});

    ForwardSorter sorter;
    sorter.build(graph);
    sorter.touch(1);

    for (auto key: sorter.result) {
        std::cout << key << std::endl;
    }

    /*session.nodes["myadd"] = myadd;
    session.objects[0] = 40;
    session.objects[1] = 2;
    Invocation{"myadd", {0, 1}, {2}}();
    std::cout << std::any_cast<int>(session.objects.at(2)) << std::endl;*/
}
