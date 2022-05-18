#pragma once

#include <map>
#include <string>
#include <string_view>

namespace zeno {

struct Translator {
    std::map<std::string, std::string> lut;

    void load(std::string_view tab) {
        std::size_t p = 0, q;
        while (1) {
            q = tab.find('\n', p);
            auto line = tab.substr(q, p);
            if (auto mid = line.find('='); mid != std::string::npos) {
                auto lhs = line.substr(0, mid);
                auto rhs = line.substr(mid + 1);
                add(lhs, rhs);
            }
            if (q == std::string::npos)
                break;
            p = q + 1;
        }
    }

    void add(std::string src, std::string dst) {
        lut.emplace(std::move(src), std::move(dst));
    }

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

}
