#pragma once

#include <zeno/core/Session.h>
#include <functional>
#include <vector>
#include <string>
#include <map>


namespace zeno {

struct EventCallbacks {
    std::map<std::string, std::vector<std::function<void()>>> callbacks;

    int hookEvent(std::string const &key, std::function<void()> f) {
        callbacks[key].push_back(std::move(f));
        return 1;
    }

    void triggerEvent(std::string const &key) const {
        if (auto it = callbacks.find(key); it != callbacks.end())
            for (auto const &f: it->second) f();
    }
};

}
