#pragma once

#include <zeno/types/IObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

struct ABCTree : IObjectClone<ABCTree> {
    std::shared_ptr<PrimitiveObject> prim;
    std::vector<std::unique_ptr<ABCTree>> children;

    inline std::shared_ptr<PrimitiveObject> getFirstPrim() const {
        if (prim) return prim;
        for (auto const &ch: children)
            if (auto p = ch->getFirstPrim())
                return p;
        return nullptr;
    }
};

}
