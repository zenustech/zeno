#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/safe_dynamic_cast.h>
#include <zeno/funcs/ObjectCodec.h>
#include <zeno/types/UserData.h>
#include <set>
#include <string>

namespace zeno {

struct DirtyChecker {
    std::set<std::string> dirts;

    void taintThisNode(std::string name) {
        dirts.insert(std::move(name));
    }

    bool amIDirty(std::string const &name) const {
        return dirts.find(name) != dirts.end();
    }
};

}
