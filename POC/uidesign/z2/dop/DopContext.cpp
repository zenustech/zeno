#include <z2/dop/DopContext.h>
#include <z2/dop/DopNode.h>
#include <algorithm>
#include <ranges>
#include <queue>


namespace z2::dop {


/*void DopContext::Ticket::wait() const {
    if (!ctx->visited.contains(node)) {
        ctx->visited.insert(node);
        node->_apply_func(ctx);
    }
}*/


bool DopDepsgraph::contains_node(DopNode *node) const {
    return nodes.contains(node);
}


void DopDepsgraph::insert_node(DopNode *node, std::set<DopNode *> &&deps) {
    nodes.emplace(node, std::move(deps));
}


void DopDepsgraph::execute() {
    struct OrderInfo {
        float new_order = 0;
        int dep_order = 0;
        float old_order = 0;

        bool operator<(OrderInfo const &that) const {
            return new_order < that.new_order
                || dep_order < that.dep_order;
        }
    };

    std::vector<DopNode *> node_list;
    std::map<DopNode *, OrderInfo> order;

    auto touch = [&] (auto touch, DopNode *node) -> OrderInfo & {
        OrderInfo ord{node->xpos, 0};
        if (order.contains(node)) {
            return order.at(node);
        } else {
            auto const &deps = nodes.at(node);
            for (auto *dep: deps) {
                auto &depord = touch(touch, dep);
                if (depord.new_order >= ord.new_order) {
                    depord.new_order = ord.new_order;
                    depord.dep_order = std::min(depord.dep_order, ord.dep_order - 1);
                }
            }
            auto it = order.emplace(node, ord).first;
            return it->second;
        }
    };

    for (auto const &[node, deps]: nodes) {
        touch(touch, node);
        node_list.push_back(node);
    }

    std::sort(node_list.begin(), node_list.end(), [&] (DopNode *p, DopNode *q) {
        return order.at(p) < order.at(q);
    });

    for (auto *node: node_list) {
        node->execute();
    }
}


}
