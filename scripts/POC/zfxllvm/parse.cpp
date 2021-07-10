#include <cstdio>
#include <cctype>
#include <string>
#include <vector>
#include <cstring>
#include <iostream>
#include <memory>
#include <set>

/* common utils */

using std::cout;
using std::endl;

template <int First, int Last, typename Lambda>
inline constexpr bool static_for(Lambda const &f) {
    if constexpr (First < Last) {
        if (f(std::integral_constant<int, First>{})) {
            return true;
        } else {
            return static_for<First + 1, Last>(f);
        }
    }
    return false;
}

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

template <size_t BufSize = 4096, class ...Ts>
std::string format(const char *fmt, Ts &&...ts) {
    char buf[BufSize];
    sprintf(buf, fmt, std::forward<Ts>(ts)...);
    return buf;
}

template <class ...Ts>
[[noreturn]] void error(const char *fmt, Ts &&...ts) {
    printf("ERROR: ");
    printf(fmt, std::forward<Ts>(ts)...);
    putchar('\n');
    exit(-1);
}

/* tokenizer */

static char opchars[] = "+-*/%=()";
static std::string opstrs[] = {"+", "-", "*", "/", "%", "=", "(", ")"};

bool is_literial_atom(std::string const &s) {
    if (!s.size()) return false;
    if (isdigit(s[0]) || s.size() > 1 && s[0] == '-' && isdigit(s[1])) {
        return true;
    }
    return false;
}

bool is_symbolic_atom(std::string const &s) {
    if (!s.size()) return false;
    if (isalpha(s[0])) {
        return true;
    }
    return false;
}

bool is_atom(std::string const &s) {
    return is_literial_atom(s) || is_symbolic_atom(s);
}

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
            error("unexpected character token: `%c`", *cp);
            break;
        }
    }
    tokens.push_back("");  // EOF sign
    return tokens;
}

/* AST parser */

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

/* IR statements */

struct Statement;

struct IRVisitor {
    virtual void visit(Statement *stmt) = 0;
};

struct Statement {
    const int id;

    explicit Statement
        ( int id_
        )
        : id(id_)
    {}

    void accept(IRVisitor *visitor) {
        visitor->visit(this);
    }

    virtual std::string print() const {
        return format("$%d = Statement");
    }
};

template <class T>
struct Stmt : Statement {
    using Statement::Statement;
};

struct UnaryOpStmt : Stmt<UnaryOpStmt> {
    std::string op;
    Statement *src;

    UnaryOpStmt
        ( int id_
        , std::string op_
        , Statement *src_
        )
        : Stmt(id_)
        , op(op_)
        , src(src_)
    {}

    virtual std::string print() const override {
        return format(
            "$%d = UnaryOp [%s] $%d"
            , id
            , op.c_str()
            , src->id
            );
    }
};

struct BinaryOpStmt : Stmt<BinaryOpStmt> {
    std::string op;
    Statement *lhs;
    Statement *rhs;

    BinaryOpStmt
        ( int id_
        , std::string op_
        , Statement *lhs_
        , Statement *rhs_
        )
        : Stmt(id_)
        , op(op_)
        , lhs(lhs_)
        , rhs(rhs_)
    {}

    virtual std::string print() const override {
        return format(
            "$%d = BinaryOp [%s] $%d $%d"
            , id
            , op.c_str()
            , lhs->id
            , rhs->id
            );
    }
};

struct AssignStmt : Stmt<AssignStmt> {
    Statement *dst;
    Statement *src;

    AssignStmt
        ( int id_
        , Statement *dst_
        , Statement *src_
        )
        : Stmt(id_)
        , dst(dst_)
        , src(src_)
    {}

    virtual std::string print() const override {
        return format(
            "$%d = Assign $%d $%d"
            , id
            , dst->id
            , src->id
            );
    }
};

struct SymbolStmt : Stmt<SymbolStmt> {
    std::string name;

    SymbolStmt
        ( int id_
        , std::string name_
        )
        : Stmt(id_)
        , name(name_)
    {}

    virtual std::string print() const override {
        return format(
            "$%d = Symbol [%s]"
            , id
            , name.c_str()
            );
    }
};

struct LiterialStmt : Stmt<LiterialStmt> {
    std::string name;

    LiterialStmt
        ( int id_
        , std::string name_
        )
        : Stmt(id_)
        , name(name_)
    {}

    virtual std::string print() const override {
        return format(
            "$%d = Literial [%s]"
            , id
            , name.c_str()
            );
    }
};

/* IR lowering */

struct IR {
    std::vector<std::unique_ptr<Statement>> stmts;

    template <class T, class ...Ts>
    T *emplace_back(Ts &&...ts) {
        auto id = stmts.size();
        auto stmt = std::make_unique<T>(id, std::forward<Ts>(ts)...);
        auto raw_ptr = stmt.get();
        stmts.push_back(std::move(stmt));
        return raw_ptr;
    }

    void print() const {
        for (auto const &s: stmts) {
            cout << s->print() << endl;
        }
    }

    void accept(IRVisitor *visitor) {
        for (auto const &stmt: stmts) {
            stmt->accept(visitor);
        }
    }

};

struct LowerAST {
    std::unique_ptr<IR> ir = std::make_unique<IR>();

    Statement *serialize(AST *ast) {
        if (0) {

        } else if (contains({"+", "-", "*", "/", "%"}, ast->token) && ast->args.size() == 2) {
            auto lhs = serialize(ast->args[0].get());
            auto rhs = serialize(ast->args[1].get());
            return ir->emplace_back<BinaryOpStmt>(ast->token, lhs, rhs);

        } else if (contains({"+", "-"}, ast->token) && ast->args.size() == 1) {
            auto src = serialize(ast->args[0].get());
            return ir->emplace_back<UnaryOpStmt>(ast->token, src);

        } else if (contains({"="}, ast->token) && ast->args.size() == 2) {
            auto dst = serialize(ast->args[0].get());
            auto src = serialize(ast->args[1].get());
            return ir->emplace_back<AssignStmt>(dst, src);

        } else if (is_symbolic_atom(ast->token) && ast->args.size() == 0) {
            return ir->emplace_back<SymbolStmt>(ast->token);

        } else if (is_literial_atom(ast->token) && ast->args.size() == 0) {
            return ir->emplace_back<LiterialStmt>(ast->token);

        } else {
            error("cannot lower AST at token: `%s` (%d args)\n",
                ast->token.c_str(), ast->args.size());
            return nullptr;
        }
    }
};

/* IR passes */

template <class T, class ...Ts>
struct Visitor : IRVisitor {
    virtual void visit(Statement *stmt) override {
        static_for<0, sizeof...(Ts)>([this, stmt] (auto i) {
            using S = std::tuple_element_t<i, std::tuple<Ts...>>;
            auto p = dynamic_cast<S *>(stmt);
            if (!p) return false;
            reinterpret_cast<T *>(this)->visit(p);
            return true;
        });
    }
};

struct DemoVisitor : Visitor
    < DemoVisitor
    , SymbolStmt
    , LiterialStmt
    > {
    void visit(SymbolStmt *stmt) {
        printf("DemoVisitor got symbol: [%s]\n", stmt->name.c_str());
    }

    void visit(LiterialStmt *stmt) {
        printf("DemoVisitor got literial: [%s]\n", stmt->name.c_str());
    }
};

/* main body */

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
    Parser parser(tokens);
    auto asts = parser.parse();
    for (auto const &a: asts) {
        print(a.get());
        cout << endl;
    }

    LowerAST lower;
    for (auto const &a: asts) {
        lower.serialize(a.get());
    }
    auto ir = std::move(lower.ir);
    ir->print();

    DemoVisitor demo;
    ir->accept(&demo);

    cout << "==============" << endl;
    return 0;
}
