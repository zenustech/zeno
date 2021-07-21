#pragma once


#include <zeno/zeno.h>
#include <cassert>


namespace zeno {

struct ContextManagedNode : INode {
    std::unique_ptr<Context> m_ctx = nullptr;

    void push_context() {
        assert(!m_ctx);
        m_ctx = std::move(graph->ctx);
        graph->ctx = std::make_unique<Context>(*m_ctx);
    }

    void pop_context() {
        assert(m_ctx);
        graph->ctx = std::move(m_ctx);
    }

    void pop_context_with_merge() {
        assert(m_ctx);
        graph->ctx->visited.merge(m_ctx->visited);
        graph->ctx = std::move(m_ctx);
    }
};

}
