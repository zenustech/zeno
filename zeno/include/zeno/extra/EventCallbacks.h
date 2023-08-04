#pragma once

#include <zeno/core/Session.h>
#include <functional>
#include <utility>
#include <vector>
#include <string>
#include <map>
#include <optional>
#include <type_traits>
#include <any>

namespace zeno {

struct EventCallbacks {
    using ParameterType = std::optional<std::any>;

    std::map<std::string, std::vector<std::function<void(ParameterType)>>> callbacks;

    int hookEvent(std::string const &key, std::function<void(ParameterType)> f) {
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

    void triggerEvent2(std::string const &key, const std::any& arg) const {
        if (auto it = callbacks.find(key); it != callbacks.end())
            for (auto const &f: it->second) f(arg);
    }
};

struct EventRegisterHelper {
    EventRegisterHelper(const std::string& event_name, std::function<void(EventCallbacks::ParameterType)> func) {
        zeno::getSession().eventCallbacks->hookEvent(event_name, std::move(func));
    }
};

}
