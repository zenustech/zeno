#include <zeno/core/INode.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/SubnetNode.h>

namespace zeno {

ZENO_API SubnetNode::SubnetNode() : subgraph(std::make_unique<Graph>())
{}

ZENO_API SubnetNode::~SubnetNode() = default;

ZENO_API void SubnetNode::apply() {
    subgraph->applyNodesToExec();
}

}
