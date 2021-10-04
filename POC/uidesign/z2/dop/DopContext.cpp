#include <z2/dop/DopContext.h>
#include <z2/dop/DopNode.h>


namespace z2::dop {

DopPromise DopContext::promise(DopNode *node, int idx) {
    promised.insert(node->name);

    return [=] () -> std::any {
        return node->outputs[idx].result;
    };
}


DopPromise DopContext::immediate(std::any val) {
    return [=] () -> std::any { return val; }
}

}
