#include "DopNode.h"
#include "DopContext.h"
#include "DopGraph.h"
#include "DopTable.h"


void DopNode::apply_func() {
    DopContext ctx;
    for (auto const &input: inputs) {
        auto val = graph->resolve_value(input.value, &applied);
        ctx.in.push_back(std::move(val));
    }
    if (applied)
        return;

    ctx.out.resize(outputs.size());
    auto func = tab.lookup(kind);
    func(&ctx);

    for (int i = 0; i < ctx.out.size(); i++) {
        outputs[i].result = std::move(ctx.out[i]);
    }
    applied = true;
}
