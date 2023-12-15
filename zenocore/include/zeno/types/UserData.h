#pragma once

#include <zeno/utils/safe_at.h>
#include <zeno/utils/safe_dynamic_cast.h>
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
    bool isa(std::string const &name) const {
        return !!dynamic_cast<T *>(get(name).get());
    }

    std::shared_ptr<IObject> get(std::string const &name, std::shared_ptr<IObject> defl) const {
        return has(name) ? get(name) : defl;
    }

    template <class T>
    std::shared_ptr<T> get(std::string const &name) const {
        return safe_dynamic_cast<T>(get(name));
    }

    template <class T>
    std::shared_ptr<T> get(std::string const &name, std::decay_t<std::shared_ptr<T>> defl) const {
        return has(name) ? get<T>(name) : defl;
    }

    template <class T>
    [[deprecated("use get2<T>(name)")]]
    T getLiterial(std::string const &name) const {
        return get2<T>(name);
    }

    template <class T>
    [[deprecated("use get2<T>(name, value)")]]
    T getLiterial(std::string const &name, T defl) const {
        return get2<T>(name, std::move(defl));
    }

    template <class T>
    T get2(std::string const &name) const {
        return objectToLiterial<T>(get(name));
    }

    template <class T>
    T get2(std::string const &name, T defl) const {
        return has(name) ? getLiterial<T>(name) : defl;
    }

    void set(std::string const &name, std::shared_ptr<IObject> value) {
        m_data[name] = std::move(value);
    }

    void merge(const UserData& name) {
        for (const auto& pair : name.m_data) {
            m_data.insert(pair);
        }
    }

    template <class T>
    [[deprecated("use set2(name, value)")]]
    void setLiterial(std::string const &name, T &&value) {
        return set2(name, std::forward<T>(value));
    }

    template <class T>
    void set2(std::string const &name, T &&value) {
        m_data[name] = objectFromLiterial(std::forward<T>(value));
    }

    void del(std::string const &name) {
        m_data.erase(name);
    }

    auto begin() const {
        return m_data.begin();
    }

    auto end() const {
        return m_data.end();
    }

    size_t size() const {
        return m_data.size();
    }

    auto begin() {
        return m_data.begin();
    }

    auto end() {
        return m_data.end();
    }

    size_t erase(const std::string& key) {
        return m_data.erase(key);
    }
};

}
