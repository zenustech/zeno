#pragma once

#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>
#include <memory>
#include <cassert>


namespace zeno {

struct ContextManagedNode : INode {
    //std::unique_ptr<Context> m_ctx = nullptr;
    bool bNewContext = false;

    void push_context() {
        /*
        assert(!m_ctx);
        std::shared_ptr<Graph> spGraph = getThisGraph();
        assert(spGraph);
        m_ctx = std::move(spGraph->ctx);
        if (m_ctx) {
            spGraph->ctx = std::make_unique<Context>(*m_ctx);
        }
        else {
            // Context may be another subgraph, which has been cleared,
            // here, we construct a temp context for calling for func.
            spGraph->ctx = std::make_unique<Context>();
            bNewContext = true;
        }
        */
    }

    /*std::unique_ptr<Context>*/
    void pop_context() {
        /*
        std::shared_ptr<Graph> spGraph = getThisGraph();
        assert(spGraph);
        if (bNewContext) {
            spGraph->ctx.reset();
            bNewContext = false;
            return nullptr;
        }
        assert(m_ctx);
        auto old_ctx = std::move(spGraph->ctx);
        spGraph->ctx = std::move(m_ctx);
        return old_ctx;
        */
    }
};

}
