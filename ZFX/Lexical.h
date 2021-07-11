#pragma once

#include "common.h"

namespace zfx {

inline char opchars[] = "+-*/%=()";
inline std::string opstrs[] = {"+", "-", "*", "/", "%", "=", "(", ")"};

inline bool is_literial_atom(std::string const &s) {
    if (!s.size()) return false;
    if (isdigit(s[0]) || s.size() > 1 && s[0] == '-' && isdigit(s[1])) {
        return true;
    }
    return false;
}

inline bool is_symbolic_atom(std::string const &s) {
    if (!s.size()) return false;
    if (isalpha(s[0]) || strchr("_$@", s[0])) {
        return true;
    }
    return false;
}

inline bool is_atom(std::string const &s) {
    return is_literial_atom(s) || is_symbolic_atom(s);
}

std::vector<std::string> tokenize(const char *cp);

}
