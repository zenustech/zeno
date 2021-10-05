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
        float xorder = 0;
        int torder = 0;

        bool operator<(OrderInfo const &that) const {
            return xorder < that.xorder || torder < that.torder;
        }
    };

    std::vector<DopNode *> node_list;
    std::map<DopNode *, OrderInfo> order;

    for (auto const &[node, deps]: nodes) {
        if (!order.contains(node)) {
            order.emplace(node, OrderInfo{node->xpos, 0});
        }
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
