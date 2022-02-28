#pragma once

#include <zeno/utils/safe_at.h>
#include <zeno/utils/memory.h>
#include <zeno/core/IObject.h>
#include <zeno/funcs/LiterialConverter.h>
#include <string>
#include <map>
#include <any>

namespace zeno {

struct UserData {
    std::map<std::string, std::shared_ptr<IObject>> m_data;

    bool has(std::string const &name) const {
        return m_data.find(name) != m_data.end();
    }

    template <class T>
    bool has(std::string const &name) const {
        auto it = m_data.find(name);
        if (it == m_data.end()) {
            return false;
        }
        return objectIsLiterial<T>(it->second);
    }

    std::shared_ptr<IObject> const &get(std::string const &name) const {
        return safe_at(m_data, name, "user data");
    }

    template <class T>
    T &getLiterial(std::string const &name) const {
        return objectToLiterial<T>(get(name));
    }

    void set(std::string const &name, std::shared_ptr<IObject> value) {
        m_data[name] = std::move(value);
    }

    template <class T>
    void setLiterial(std::string const &name, T &&value) {
        m_data[name] = objectFromLiterial(std::forward<T>(value));
    }

    void del(std::string const &name) {
        m_data.erase(name);
    }
};

}
