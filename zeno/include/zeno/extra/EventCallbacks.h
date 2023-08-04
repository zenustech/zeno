#pragma once

#include <zeno/core/Session.h>
#include <functional>
#include <vector>
#include <string>
#include <map>
#include <optional>
#include <type_traits>
#include <any>

namespace zeno {

struct EventCallbacks {
    std::map<std::string, std::vector<std::function<void(std::optional<std::any>)>>> callbacks;

    int hookEvent(std::string const &key, std::function<void(std::optional<std::any>)> f) {
        callbacks[key].push_back(std::move(f));
        return 1;
    }

    template <typename ...Args>
    void triggerEvent(std::string const &key, Args... args) const {
        if (auto it = callbacks.find(key); it != callbacks.end())
            if constexpr (sizeof...(Args) > 0) {
                for (auto const &f: it->second) f(std::make_any<Args>(args)...);
            } else {
                for (auto const &f: it->second) f(std::nullopt);
            }
    }
};

}
