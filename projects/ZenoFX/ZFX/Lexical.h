#pragma once

#include <zfx/utils.h>

namespace zfx {

//inline char opchars[] = "+-*/%=(),.;<>!&|^?:";
//
enum class Op {
    '+';
    '-'

};

/*inline std::set<std::string> opstrs = {
    "(", ")", ",", ".", ";",
    "+", "-", "*", "/", "%", "=",
    "+=", "-=", "*=", "/=", "%=",
    "==", "!=", "<", "<=", ">", ">=",
    "&", "&!", "|", "^", "!", "?", ":",
    };
*/
enum class Seprator {
    '(',


};

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

inline int swizzle_from_char(char c) {
    if ('x' <= c && c <= 'z') {
        return c - 'x';
    } else if (c == 'w') {
        return 3;
    } else if ('0' <= c && c <= '9') {
        return c - '0';
    } else if ('a' <= c && c <= 'v') {
        return c - 'a' + 10;
    } else {
        return -1;
    }
}

std::vector<std::string> tokenize(const char *cp);

}
