#pragma once


#include <zeno/dop/Node.h>
#include <vector>
#include <set>


ZENO_NAMESPACE_BEGIN
namespace dop {


struct SceneGraph {
    std::set<std::unique_ptr<Node>> nodes;

    std::vector<Node *> visibleNodes() const;
};


}
ZENO_NAMESPACE_END
