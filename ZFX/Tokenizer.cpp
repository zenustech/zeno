#include "Lexical.h"

namespace zfx {

std::vector<std::string> tokenize(const char *cp) {
    std::vector<std::string> tokens;
    while (1) {
        for (; *cp && isspace(*cp); cp++);
        if (!*cp)
            break;

        if (*cp == '#') {
            for (; *cp && *cp != '\n'; cp++);

        } else if (isalpha(*cp) || strchr("_$@", *cp)) {
            std::string res;
            res += *cp++;
            for (; isalnum(*cp) || *cp && strchr("_$@", *cp); cp++)
                res += *cp;
            tokens.push_back(res);

        } else if (isdigit(*cp) || *cp == '-' && isdigit(cp[1])) {
            std::string res;
            res += *cp++;
            for (; isdigit(*cp) || *cp == '.'; cp++)
                res += *cp;
            tokens.push_back(res);

        } else if (strchr(opchars, *cp)) {
            std::string res;
            do {
                res += *cp++;
            } while (contains(opstrs, res + *cp));
            tokens.push_back(res);

        } else {
            error("unexpected character token: `%c`", *cp);
            break;
        }
    }
    tokens.push_back("");  // EOF sign
    return tokens;
}

}
