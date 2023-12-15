#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/types/NumericObject.h>
#include <string>
#include <map>
#include <any>

namespace zeno {

struct ShaderNode;

struct ShaderObject : IObjectClone<ShaderObject> {
    std::shared_ptr<ShaderNode> node;

    explicit ShaderObject(ShaderNode *node) : node(node->clone()) {}
};

}
