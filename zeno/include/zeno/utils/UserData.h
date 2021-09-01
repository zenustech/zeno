#pragma once

#include <zeno/utils/any.h>
#include <string>
#include <map>

namespace zeno {

struct UserData {
    std::map<std::string, zany> m_data;

    template <class T>
    inline T &get(std::string const &name) {
        auto it = m_data.find(name);
        if (it == m_data.end()) {
            auto ptr = std::make_shared<T>();
            auto raw_ptr = ptr.get();
            m_data[name] = std::move(ptr);
            return *raw_ptr;
        }
        return *smart_any_cast<std::shared_ptr<T>>(it->second);
    }

    template <class T>
    inline void set(std::string const &name, T const &value) {
        m_data[name] = std::make_shared<T>(value);
    }

    template <class T>
    inline void set(std::string const &name, T &&value) {
        m_data[name] = std::make_shared<T>(std::move(value));
    }

    inline void del(std::string const &name) {
        m_data.erase(name);
    }
};

}
