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
    ShaderNode *node;
    std::any extra_data;

    ZENO_API explicit ShaderObject(ShaderNode *node) : node(std::move(node)) {}
};

}
