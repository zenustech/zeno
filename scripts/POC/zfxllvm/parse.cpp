#include <cstdio>
#include <cctype>
#include <string>
#include <vector>
#include <cstring>
#include <iostream>

using std::cout;
using std::endl;

static inline char opchars[] = "+-*/%=";

std::vector<std::string> tokenize(const char *cp) {
    std::vector<std::string> tokens;
    while (true) {
        for (; *cp && isspace(*cp); cp++);
        if (!*cp)
            break;

        if (isalpha(*cp) || strchr("_$@", *cp)) {
            std::string res;
            for (; isalnum(*cp) || *cp && strchr("_$@", *cp); cp++)
                res += *cp;
            tokens.push_back(res);

        } else if (isdigit(*cp) || *cp == '-' && isdigit(cp[1])) {
            std::string res;
            for (; isdigit(*cp) || *cp && strchr(".e-", *cp); cp++)
                res += *cp;
            tokens.push_back(res);

        } else if (strchr(opchars, *cp)) {
            std::string res;
            for (; strchr(opchars, *cp); cp++)
                res += *cp;
            tokens.push_back(res);

        } else {
            printf("unexpected token: `%c`", cp);
            break;
        }
    }
    return tokens;
}

int main() {
    std::string code("pos = 1 * 3");
    auto tokens = tokenize(code.c_str());
    for (auto t: tokens) {
        cout << t << endl;
    }
    return 0;
}
