#include <cstdio>
#include <cctype>
#include <string>
#include <vector>
#include <cstring>
#include <iostream>
#include <memory>

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

struct AST {
    std::string token;
    std::vector<AST *> args;

    explicit AST(std::string const &token, std::vector<AST *> const &args = {})
        : token(token), args(args) {}
};

struct Parser {
    std::vector<std::string> tokens;
    typename std::vector<std::string>::iterator token;
    std::vector<std::unique_ptr<AST>> asts;

    Parser(std::vector<std::string> const &tokens_) : tokens(tokens_) {
        token = tokens.begin();
    }

    void parse_atom() {
        auto ast = std::make_unique<AST>(*token++);
        asts.push_back(std::move(ast));
    }

    void parse() {
        parse_atom();
    }
};

void print(AST *ast) {
    if (ast->args.size())
        cout << '(';
    cout << ast->token;
    for (auto const &a: ast->args) {
        cout << ' ';
        print(a);
    }
    if (ast->args.size())
        cout << ')';
}

int main() {
    std::string code("pos = 1 * 3");
    auto tokens = tokenize(code.c_str());
    Parser p(tokens);
    p.parse();
    for (auto const &a: p.asts) {
        print(a.get());
    }
    cout << endl;
    return 0;
}
