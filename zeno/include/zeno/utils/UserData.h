#pragma once

#include <zeno/utils/any.h>
#include <string>
#include <map>

namespace zeno {

struct UserData {
    std::map<std::string, zany> m_data;

    template <class T>
    T get(std::string const &name) {
        auto it = m_data.find(name);
        if (it == m_data.end()) {
            return (m_data[name] = T{});
        }
        return smart_any_cast<T>(it->second);
    }

    template <class T>
    void set(std::string const &name, T const &value) {
        m_data[name] = value;
    }
};

}
