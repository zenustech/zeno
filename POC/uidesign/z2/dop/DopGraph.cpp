#include <z2/dop/DopGraph.h>
#include <z2/dop/DopNode.h>


namespace z2::dop {


DopNode *DopGraph::add_node(std::string kind) {
    auto p = std::make_unique<DopNode>();
    p->graph = this;
    auto name = _determine_name(kind);
    p->kind = kind;
    p->name = name;
    auto raw = p.get();
    nodes.emplace(name, std::move(p));
    return raw;
}


std::string DopGraph::_determine_name(std::string kind) {
    for (int i = 1; i <= 256; i++) {
        auto name = kind + std::to_string(i);
        if (!nodes.contains(name)) {
            return name;
        }
    }
    return kind + std::to_string(std::rand()) + 'a';
}

bool DopGraph::remove_node(DopNode *node) {
    for (auto const &[k, n]: nodes) {
        if (n.get() == node) {
            nodes.erase(k);
            return true;
        }
    }
    return false;
}

void DopGraph::serialize(std::ostream &ss) const {
    for (auto const &[k, node]: nodes) {
        node->serialize(ss);
        void(ss << '\n');
    }
}

void DopGraph::set_node_input
    ( DopNode *to_node
    , int to_socket_index
    , DopNode *from_node
    , int from_socket_index
    )
{
    auto const &from_socket = from_node->outputs.at(from_socket_index);
    auto &to_socket = to_node->inputs.at(to_socket_index);
    auto refid = '@' + from_node->name + ':' + from_socket.name;
    to_socket.value = refid;
}


void DopGraph::remove_node_input
    ( DopNode *to_node
    , int to_socket_index
    )
{
    auto &to_socket = to_node->inputs.at(to_socket_index);
    to_socket.value = {};
}


std::any DopGraph::resolve_value(std::string expr, DopContext *visited) {
    if (expr[0] == '@') {
        auto i = expr.find(':');
        auto node_n = expr.substr(1, i - 1);
        auto socket_n = expr.substr(i + 1);
        auto *node = nodes.at(node_n).get();
        auto val = node->get_output_by_name(socket_n, visited);
        return val;

    } else if (!expr.size()) {
        return {};

    } else if (std::strchr("0123456789+-.", expr[0])) {
        if (expr.find('.') != std::string::npos) {
            return std::stof(expr);
        } else {
            return std::stoi(expr);
        }

    } else {
        return expr;
    }
}


}  // namespace z2::dop
