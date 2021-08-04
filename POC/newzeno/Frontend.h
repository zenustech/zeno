#pragma once

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


struct Graph {
    struct Node {
        std::string name;
        std::vector<std::pair<int, int>> inputs;
        int num_outputs = 0;
    };
    std::vector<Node> nodes;
};


struct ForwardSorter {
    std::set<int> visited;
    std::map<int, std::vector<int>> links;
    std::vector<int> result;

    Graph const &graph;

    ForwardSorter(Graph const &graph) : graph(graph) {
        for (int dst_node = 0; dst_node < graph.nodes.size(); dst_node++) {
            auto &link = links[dst_node];
            auto const &node = graph.nodes.at(dst_node);
            for (auto const &[src_node, src_sock]: node.inputs) {
                if (src_node != -1)
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

    auto linearize() {
        int lutid = 0;
        auto ir = std::make_unique<IRBlock>();
        std::map<std::pair<int, int>, int> lut;
        for (auto nodeid: result) {
            auto const &node = graph.nodes.at(nodeid);
            Invocation invo;
            invo.node_name = node.name;
            for (auto const &source: node.inputs) {
                if (source.first != -1)
                    invo.inputs.push_back(lut.at(source));
            }
            for (int sockid = 0; sockid < node.num_outputs; sockid++) {
                auto id = lutid++;
                lut[std::make_pair(nodeid, sockid)] = id;
                invo.outputs.push_back(id);
            }
            ir->invos.push_back(invo);
        }
        return ir;
    }
};


void print_invocation(Invocation const &invo) {
    std::cout << "[";
    bool had = false;
    for (auto const &output: invo.outputs) {
        if (had) std::cout << ", ";
        else had = true;
        std::cout << output;
    }
    std::cout << "] = ";
    std::cout << invo.node_name;
    std::cout << "(";
    had = false;
    for (auto const &input: invo.inputs) {
        if (had) std::cout << ", ";
        else had = true;
        std::cout << input;
    }
    std::cout << ");\n";
}
