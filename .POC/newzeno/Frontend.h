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
        root_block = std::make_unique<statement::IRBlock>();
        current_block = root_block.get();
    }

    std::unique_ptr<statement::IRBlock> get_root() {
        return std::move(root_block);
    }

    void require(int key) {
        if (auto it = visited.find(key); it != visited.end()) {
            return;
        }
        visited.insert(key);

        auto stmt = parse_node(key);
        current_block->stmts.emplace_back(std::move(stmt));
    }

    void require_all_inputs(int key) {
        if (auto it = links.find(key); it != links.end()) {
            for (auto const &source: it->second) {
                require(source);
            }
        }
    }

    int lutid = 0;
    std::map<std::pair<int, int>, int> lut;
    statement::IRBlock *current_block = nullptr;
    std::unique_ptr<statement::IRBlock> root_block;

    int lut_entry(int nodeid, int sockid) {
        auto id = lutid++;
        lut[std::make_pair(nodeid, sockid)] = id;
        return id;
    }

    auto lut_require(std::pair<int, int> const &key) {
        require(key.first);
        return lut.at(key);
    }

    std::unique_ptr<statement::Statement> parse_node(int nodeid) {
        auto const &node = graph.nodes.at(nodeid);

        if (node.name == "if") {
            auto stmt = std::make_unique<statement::StmtIfBlock>();
            stmt->input_cond = lut_require(node.inputs.at(0));
            stmt->input_true = lut_require(node.inputs.at(1));
            stmt->input_false = lut_require(node.inputs.at(2));
            stmt->output = lut_entry(nodeid, 0);
            return std::move(stmt);
        }

        require_all_inputs(nodeid);

        if (node.name == "value") {
            auto stmt = std::make_unique<statement::StmtValue>();
            stmt->output = lut_entry(nodeid, 0);
            stmt->value = node.parameter;
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
};

}
