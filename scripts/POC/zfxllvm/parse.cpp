#include <cstdio>
#include <cctype>
#include <string>
#include <vector>
#include <cstring>
#include <iostream>
#include <memory>

using std::cout;
using std::endl;

template <class T>
struct copiable_unique_ptr : std::unique_ptr<T> {
    using std::unique_ptr<T>::unique_ptr;
    using std::unique_ptr<T>::operator=;

    copiable_unique_ptr(std::unique_ptr<T> &&o)
        : std::unique_ptr<T>(std::move(o)) {
    }

    copiable_unique_ptr(copiable_unique_ptr const &o)
        : std::unique_ptr<T>(std::make_unique<T>(
            static_cast<T const &>(*o))) {
    }

    operator std::unique_ptr<T> &() { return *this; }
    operator std::unique_ptr<T> const &() const { return *this; }
};


template <class T>
copiable_unique_ptr(std::unique_ptr<T> &&o) -> copiable_unique_ptr<T>;

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
    using Ptr = copiable_unique_ptr<AST>;

    std::string token;
    std::vector<AST::Ptr> args;

    explicit AST
        ( std::string const &token_
        , std::vector<AST::Ptr> const &args_ = {}
        )
        : token(std::move(token_))
        , args(std::move(args_))
        {}
};

template <class ...Ts>
AST::Ptr make_ast(Ts &&...ts) {
    return std::make_unique<AST>(ts...);
}

struct Parser {
    std::vector<std::string> tokens;
    typename std::vector<std::string>::iterator token;
    std::vector<AST::Ptr> asts;

    Parser(std::vector<std::string> const &tokens_) : tokens(tokens_) {
        token = tokens.begin();
    }

    AST::Ptr parse_atom() {
        return make_ast(*token++);
    }

    AST::Ptr parse() {
        return parse_atom();
    }
};

void print(AST *ast) {
    if (ast->args.size())
        cout << '(';
    cout << ast->token;
    for (auto const &a: ast->args) {
        cout << ' ';
        print(a.get());
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
