#pragma once

#include <zeno/core/Session.h>
#include <functional>
#include <map>


namespace zeno {

struct Initializer {
    std::vector<std::function<void()>> callbacks;

    int defInit(std::function<void()> f) {
        callbacks.push_back(std::move(f));
        return 1;
    }

    void doInit() {
        for (auto const &f: callbacks) f();
    }
};

}
