#pragma once

#include <zeno/utils/any.h>
#include <string>
#include <map>

namespace zeno {

struct UserData {
    std::map<std::string, zany> m_data;

    ZENO_API zany &at(std::string const &name);

    template <class T>
    T get(std::string const &name) {
        return smart_any_cast<T>(at(name));
    }

    template <class T>
    T set(std::string const &name, T const &value) {
        getUserData(name) = value;
    }
};

}
