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
    void triggerEvent(std::string const &key, Args&&... args) const {
        if (auto it = callbacks.find(key); it != callbacks.end()) {
            static constexpr size_t SizeOfArgs = sizeof...(Args);
            using Tuple = std::tuple<Args...>;
            if constexpr (SizeOfArgs > 0) {
                for (auto const &f : it->second) f(std::make_tuple<Tuple>(std::forward<Args...>(args)...));
            } else {
                for (auto const &f : it->second) f(std::nullopt);
            }
        }
    }

    void triggerEvent(std::string const &key, const std::any& arg) const {
        if (auto it = callbacks.find(key); it != callbacks.end())
            for (auto const &f: it->second) f(arg);
    }
};

}
