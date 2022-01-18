#pragma once

#include <zeno/utils/Any.h>
#include <zeno/utils/safe_at.h>
#include <string>
#include <map>

namespace zeno {

struct UserData {
    std::map<std::string, Any> m_data;

    inline bool has(std::string const &name) const {
        return m_data.find(name) != m_data.end();
    }

    template <class T>
    inline bool has(std::string const &name) const {
        auto it = m_data.find(name);
        if (it == m_data.end()) {
            return false;
        }
        return silent_any_cast<std::shared_ptr<T>>(it->second).has_value();
    }

    template <class T = Any>
    inline T &get(std::string const &name) const {
        return *safe_any_cast<std::shared_ptr<T>>(safe_at(m_data, name, "user data"));
    }

    template <class T = Any>
    inline T &get(std::string const &name) {
        auto it = m_data.find(name);
        if (it == m_data.end()) {
            auto ptr = std::make_shared<T>();
            auto raw_ptr = ptr.get();
            m_data[name] = std::move(ptr);
            return *raw_ptr;
        }
        return *safe_any_cast<std::shared_ptr<T>>(it->second);
    }

    template <class T = Any>
    inline void set(std::string const &name, T const &value) {
        m_data[name] = std::make_shared<T>(value);
    }

    template <class T = Any>
    inline void set(std::string const &name, T &&value) {
        m_data[name] = std::make_shared<T>(std::move(value));
    }

    inline void del(std::string const &name) {
        m_data.erase(name);
    }
};

}
