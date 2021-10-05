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
    order.push_back(node);
}


struct compare_op {
    bool operator()(DopNode *p, DopNode *q) const {
        return p->xorder > q->xorder;
    }
};

std::priority_queue<DopNode *, std::vector<DopNode *>, compare_op> que;


void DopDepsgraph::execute() {
    for (auto const &[node, deps]: nodes) {
    }
}


}
