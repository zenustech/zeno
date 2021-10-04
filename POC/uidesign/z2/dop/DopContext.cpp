#include <z2/dop/DopContext.h>
#include <z2/dop/DopNode.h>


namespace z2::dop {


void DopContext::Ticket::wait() const {
    if (!ctx->visited.contains(node))
        node->_apply_func(ctx);
}


}
