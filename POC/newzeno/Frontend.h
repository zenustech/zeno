#pragma once

#include "Container.h"
#include "Statement.h"


namespace zeno::v2::frontend {

struct Graph {
    struct Node {
        std::string name;
        std::vector<std::pair<int, int>> inputs;
        int num_outputs = 0;
        container::any parameter{};
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

    void require(int key) {
        if (auto it = visited.find(key); it != visited.end()) {
            return;
        }
        visited.insert(key);
        if (auto it = links.find(key); it != links.end()) {
            for (auto const &source: it->second) {
                require(source);
            }
        }
        result.push_back(key);
    }

    int lutid = 0;
    std::map<std::pair<int, int>, int> lut;
    statement::IRBlock *current_block;

    int lut_entry(int nodeid, int sockid) {
        auto id = lutid++;
        lut[std::make_pair(nodeid, sockid)] = id;
        return id;
    }

    std::unique_ptr<statement::Statement>
        parse_node(int nodeid, Graph::Node const &node) {
        if (node.name == "value") {
            auto stmt = std::make_unique<statement::StmtValue>();
            stmt->output = lut_entry(nodeid, 0);
            stmt->value = node.parameter;
            return stmt;
        }
        if (node.name == "if") {
            auto stmt = std::make_unique<statement::StmtIfBlock>();
            stmt->cond_input = lut.at(node.inputs.at(0));
            stmt->block = std::make_unique<statement::IRBlock>();
            current_block = stmt->block.get();
            return stmt;
        }

        auto stmt = std::make_unique<statement::StmtCall>();
        stmt->node_name = node.name;
        for (auto const &source: node.inputs) {
            if (source.first != -1)
                stmt->inputs.push_back(lut.at(source));
        }
        for (int sockid = 0; sockid < node.num_outputs; sockid++) {
            stmt->outputs.push_back(lut_entry(nodeid, sockid));
        }
        return stmt;
    }

    auto linearize() {
        auto ir = std::make_unique<statement::IRBlock>();
        for (auto nodeid: result) {
            auto const &node = graph.nodes.at(nodeid);
            auto stmt = parse_node(nodeid, node);
            ir->stmts.emplace_back(std::move(stmt));
        }
        return ir;
    }
};

}
