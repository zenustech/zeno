#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>

namespace zeno {

struct TreeNode : INode {
    ZENO_API void settle(std::string output,
                         std::vector<std::string> const &inputs,
                         std::vector<std::string> const &params);
};

}
