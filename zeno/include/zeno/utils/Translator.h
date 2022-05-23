#pragma once

#include <map>
#include <string>
#include <string_view>

namespace zeno {

struct TranslatorFwd {
    std::map<std::string, std::string> lut;

    std::string const &t(std::string const &s) const {
        if (auto it = lut.find(s); it != lut.end()) {
            return it->second;
        } else {
            return s;
        }
    }

    const char *t(const char *s) const {
        if (auto it = lut.find(s); it != lut.end()) {
            return it->second.c_str();
        } else {
            return s;
        }
    }

    template <class S, decltype(S::fromStdString(std::string()),
                                std::declval<S>().toStdString(),
                                0) = 0>
    S t(S s) const {
        if (auto it = lut.find(s.toStdString()); it != lut.end()) {
            return S::fromStdString(it->second);
        } else {
            return s;
        }
    }
};

struct Translator : TranslatorFwd {
    TranslatorFwd m_untrans;

    void load(std::string_view tab);

    using TranslatorFwd::t;

    template <class Arg>
    auto ut(Arg &&arg) const {
        return m_untrans.t(std::forward<Arg>(arg));
    }
};

}
