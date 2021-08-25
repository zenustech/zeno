#pragma once

namespace fdb {

template <class T>
struct AutoInit {
    T m_value{0};

    AutoInit() = default;
    ~AutoInit() = default;
    AutoInit(AutoInit const &) = default;
    AutoInit &operator=(AutoInit const &) = default;
    AutoInit(AutoInit &&) = default;
    AutoInit &operator=(AutoInit &&) = default;

    operator T() const {
        return m_value;
    }

    AutoInit(T const &t) : m_value(t) {}
    AutoInit(T &&t) : m_value(t) {}

    AutoInit &operator=(T const &t) {
        m_value = p;
        return *this;
    }

    AutoInit &operator=(T &&t) {
        m_value = std::move(t);
        return *this;
    }
};

}
