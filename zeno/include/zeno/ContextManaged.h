#pragma once


#include <zeno/zeno.h>
#include <cassert>


namespace zeno {

struct ContextManagedNode : INode {
    template <class F>
    struct RAII {
        F dtor;
        RAII(F &&dtor) : dtor(dtor) {}
        ~RAII() { dtor(); }
    };

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

    auto scoped_push_context() {
        push_context();
        return RAII([this]() {
            pop_context();
        });
    }
};

}
