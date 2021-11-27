#pragma once

#include <zeno/core/IObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

struct ABCTree : IObjectClone<ABCTree> {
    std::string name;
    std::shared_ptr<PrimitiveObject> prim;
    std::vector<std::shared_ptr<ABCTree>> children;

    template <class Func>
    bool visitPrims(Func const &func) const {
        if constexpr (std::is_void_v<std::invoke_result_t<Func,
                      std::shared_ptr<PrimitiveObject> const &>>) {
            if (prim)
                func(prim);
            for (auto const &ch: children)
                ch->visitPrims(func);
        } else {
            if (prim)
                if (!func(prim))
                    return false;
            for (auto const &ch: children)
                if (!ch->visitPrims(func))
                    return false;
            return true;
        }
    }
};

}
