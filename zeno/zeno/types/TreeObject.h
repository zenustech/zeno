#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/types/NumericObject.h>
#include <string>
#include <map>

namespace zeno {

struct TreeNode;

struct TreeObject : IObjectClone<TreeObject> {
    std::vector<std::shared_ptr<IObject>> inputs;
    std::map<std::string, std::shared_ptr<IObject>> params;
};

}
