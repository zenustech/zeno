#include <zeno/dop/SceneGraph.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


std::vector<Node *> SceneGraph::visibleNodes() {
    std::vector<Node *> res;
    for (auto const &node: nodes) {
        res.push_back(node.get());
    }
    return res;
}


}
ZENO_NAMESPACE_END
