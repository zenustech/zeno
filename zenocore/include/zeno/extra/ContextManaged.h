#pragma once

#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>
#include <memory>
#include <cassert>


namespace zeno {

struct ContextManagedNode : INode {
    std::unique_ptr<Context> m_ctx = nullptr;
    bool bNewContext = false;

    void push_context() {
        assert(!m_ctx);
        m_ctx = std::move(graph->ctx);
        if (m_ctx) {
            graph->ctx = std::make_unique<Context>(*m_ctx);
        }
        else {
            // Context may be another subgraph, which has been cleared,
            // here, we construct a temp context for calling for func.
            graph->ctx = std::make_unique<Context>();
            bNewContext = true;
        }
    }

    std::unique_ptr<Context> pop_context() {
        if (bNewContext) {
            graph->ctx.reset();
            bNewContext = false;
            return nullptr;
        }
        assert(m_ctx);
        auto old_ctx = std::move(graph->ctx);
        graph->ctx = std::move(m_ctx);
        return old_ctx;
    }
};

}
