#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>

namespace zeno {

struct TreeNode : std::enable_shared_from_this<TreeNode>, INode {
    ZENO_API void settle(std::string output,
                         std::vector<std::string> const &inputs,
                         std::vector<std::string> const &params);
};

}
