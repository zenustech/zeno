#pragma once

#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>
#include <memory>
#include <cassert>


namespace zeno {

struct ContextManagedNode : INode {
    std::unique_ptr<Context> m_ctx = nullptr;

    void push_context() {
        assert(!m_ctx);
        m_ctx = std::move(graph->ctx);
        graph->ctx = std::make_unique<Context>(*m_ctx);
    }

    std::unique_ptr<Context> pop_context() {
        assert(m_ctx);
        auto old_ctx = std::move(graph->ctx);
        graph->ctx = std::move(m_ctx);
        return old_ctx;
    }
};

}
