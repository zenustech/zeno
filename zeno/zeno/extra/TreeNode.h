#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>

namespace zeno {

struct TreeNode : std::enable_shared_from_this<TreeNode>, INode {
    ZENO_API virtual void apply() override;
    ZENO_API virtual int determineType() = 0;
    ZENO_API static int determineTypeOf(IObject *object);

    ZENO_API TreeNode();
    ZENO_API ~TreeNode();
};

}
