#include <z2/dop/DopContext.h>
#include <z2/dop/DopNode.h>
#include <ranges>


namespace z2::dop {


/*void DopContext::Ticket::wait() const {
    if (!ctx->visited.contains(node)) {
        ctx->visited.insert(node);
        node->_apply_func(ctx);
    }
}*/


bool DopDepsgraph::insert_node(DopNode *node) {
    if (!nodes.contains(node)) {
        nodes.insert(node);
        order.push_back(node);
        return true;
    } else {
        return false;
    }
}


void DopDepsgraph::execute() {
#ifndef __CLANGD__
    for (auto *node: order | std::views::reverse) {
        node->execute();
    }
#endif
}


}
