#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/types/NumericObject.h>
#include <string>
#include <map>
#include <any>

namespace zeno {

struct TreeNode;

struct TreeObject : IObjectClone<TreeObject> {
    TreeNode *node;
    std::any extra_data;

    ZENO_API explicit TreeObject(TreeNode *node) : node(std::move(node)) {}
};

}
