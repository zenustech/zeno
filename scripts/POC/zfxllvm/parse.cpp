#include <cstdio>
#include <cctype>
#include <string>
#include <vector>
#include <cstring>
#include <iostream>
#include <memory>
#include <set>

using std::cout;
using std::endl;

template <class T>
struct copiable_unique_ptr : std::unique_ptr<T> {
    using std::unique_ptr<T>::unique_ptr;
    using std::unique_ptr<T>::operator=;

    copiable_unique_ptr &operator=(copiable_unique_ptr const &o) {
        std::unique_ptr<T>::operator=(std::unique_ptr<T>(
            std::make_unique<T>(static_cast<T const &>(*o))));
        return *this;
    }

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

template <class T>
bool contains(std::set<T> const &list, T const &value) {
    return list.find(value) != list.end();
}

static inline char opchars[] = "+-*/%=()";

std::vector<std::string> tokenize(const char *cp) {
    std::vector<std::string> tokens;
    while (1) {
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
            res += *cp++;
            tokens.push_back(res);

        } else {
            printf("unexpected character token: `%c`", *cp);
            break;
        }
    }
    tokens.push_back("");  // EOF sign
    return tokens;
}

using Iter = typename std::vector<std::string>::iterator;

struct AST {
    using Ptr = copiable_unique_ptr<AST>;

    Iter iter;
    std::string token;
    std::vector<AST::Ptr> args;

    explicit AST
        ( std::string const &token_
        , Iter iter_
        , std::vector<AST::Ptr> const &args_ = {}
        )
        : token(std::move(token_))
        , iter(std::move(iter_))
        , args(std::move(args_))
        {}
};

AST::Ptr make_ast(std::string const &token, Iter iter, std::vector<AST::Ptr> const &args = {}) {
    return std::make_unique<AST>(token, iter, args);
}

bool is_atom(std::string const &s) {
    if (!s.size()) return false;
    if (isdigit(s[0]) || s.size() > 1 && s[0] == '-' && isdigit(s[1])) {
        return true;
    }
    if (isalpha(s[0])) {
        return true;
    }
    return false;
}

struct Parser {
    std::vector<std::string> tokens;

    explicit Parser
        ( std::vector<std::string> const &tokens_
        )
        : tokens(tokens_)
    {
    }

    AST::Ptr parse_atom(Iter iter) {
        if (auto s = *iter; is_atom(s)) {
            return make_ast(s, iter + 1);
        }
        return nullptr;
    }

    AST::Ptr parse_operator(Iter iter, std::set<std::string> const &allows) {
        if (auto s = *iter; contains(allows, s)) {
            return make_ast(s, iter + 1);
        }
        return nullptr;
    }

    AST::Ptr parse_factor(Iter iter) {
        if (auto a = parse_atom(iter); a) {
            return a;
        }
        if (auto ope = parse_operator(iter, {"+", "-"}); ope) {
            if (auto rhs = parse_factor(ope->iter); rhs) {
                return make_ast(ope->token, rhs->iter, {std::move(rhs)});
            }
        }
        if (auto ope = parse_operator(iter, {"("}); ope) {
            if (auto rhs = parse_expr(ope->iter); rhs) {
                if (auto ket = parse_operator(rhs->iter, {")"}); ket) {
                    rhs->iter = ket->iter;
                    return rhs;
                }
            }
        }
        return nullptr;
    }

    AST::Ptr parse_term(Iter iter) {
        if (auto lhs = parse_factor(iter); lhs) {
            while (1) if (auto ope = parse_operator(lhs->iter, {"*", "/", "%"}); ope) {
                if (auto rhs = parse_factor(ope->iter); rhs) {
                    lhs = make_ast(ope->token, rhs->iter, {std::move(lhs), std::move(rhs)});
                }
            } else break;
            return lhs;
        }
        return nullptr;
    }

    AST::Ptr parse_expr(Iter iter) {
        if (auto lhs = parse_term(iter); lhs) {
            while (1) if (auto ope = parse_operator(lhs->iter, {"+", "-"}); ope) {
                    if (auto rhs = parse_term(ope->iter); rhs) {
                        lhs = make_ast(ope->token, rhs->iter, {std::move(lhs), std::move(rhs)});
                    }
            } else break;
            return lhs;
        }
        return nullptr;
    }

    AST::Ptr parse_stmt(Iter iter) {
        if (auto lhs = parse_atom(iter); lhs) {
            if (auto ope = parse_operator(lhs->iter, {"="}); ope) {
                if (auto rhs = parse_expr(ope->iter); rhs) {
                    return make_ast(ope->token, rhs->iter, {std::move(lhs), std::move(rhs)});
                }
            }
        }
        return nullptr;
    }

    auto parse() {
        std::vector<AST::Ptr> asts;
        Iter iter = tokens.begin();
        while (iter != tokens.end()) {
            auto p = parse_stmt(iter);
            if (!p) break;
            iter = p->iter;
            asts.push_back(std::move(p));
        }
        return asts;
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
    std::string code("pos = 1 + (2 + x*4) * 3");
    cout << code << endl;

    cout << "==============" << endl;
    auto tokens = tokenize(code.c_str());
    for (auto const &t: tokens) {
        cout << t << ' ';
    }
    cout << endl;

    cout << "==============" << endl;
    Parser p(tokens);
    auto asts = p.parse();
    for (auto const &a: asts) {
        print(a.get());
        cout << endl;
    }

    cout << "==============" << endl;
    return 0;
}
