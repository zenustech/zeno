#pragma once

#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/types/NumericObject.h>
#include <string>
#include <map>
#include <any>

namespace zeno {

struct ZenMatNode;

struct ZenMatObject : IObjectClone<ZenMatObject> {
    ZenMatNode *node;
    std::any extra_data;

    ZENO_API explicit ZenMatObject(ZenMatNode *node) : node(std::move(node)) {}
};

}
