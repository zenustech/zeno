#include <vector>
#include <type_traits>
#include <magic_enum.hpp>
#include <iostream>
#include <cassert>
#include <sstream>
#include <cstring>
#include <cctype>
#include <memory>
#include <array>
#include <stack>
#include <set>

struct Parser {
    std::istringstream sin;

    Parser(std::string const &code) : sin(code) {}

    struct Token {
        enum class Type {
            op, mem, reg, imm, none,
        } type;
        std::string ident;

        Token(Type type, std::string const &ident) : type(type), ident(ident) {}

        bool is_op(std::set<std::string> const &list) {
            return type == Type::op && list.find(ident) != list.end();
        }
    };

    std::vector<Token> tokens;
    decltype(tokens.begin()) token;

    void init_parse() {
        tokens.emplace_back(Token::Type::none, "EOF");
        token = tokens.begin();
    }

    struct AST {
        Token token;
        std::vector<std::unique_ptr<AST>> args;

        explicit AST
            ( Token token
            , std::vector<std::unique_ptr<AST>> args = {}
            )
            : token(std::move(token))
            , args(std::move(args))
            {}

        explicit AST
            ( Token token
            , std::unique_ptr<AST> lhs
            )
            : token(std::move(token))
            {
                args.push_back(std::move(lhs));
            }

        explicit AST
            ( Token token
            , std::unique_ptr<AST> lhs
            , std::unique_ptr<AST> rhs
            )
            : token(std::move(token))
            {
                args.push_back(std::move(lhs));
                args.push_back(std::move(rhs));
            }

        void print(std::string const &indent = "") const {
            printf("%s%s", indent.c_str(), token.ident.c_str());
            if (args.size() == 0) {
                printf("\n");
                return;
            }
            printf("(\n", indent.c_str());
            for (auto const &arg: args) {
                arg->print(indent + "  ");
            }
            printf("%s)\n", indent.c_str());
        }
    };

    std::stack<std::unique_ptr<AST>> ast_nodes;

    std::unique_ptr<AST> pop_ast() {
        auto ptr = std::move(ast_nodes.top());
        ast_nodes.pop();
        return ptr;
    }

    template <class ...Ts>
    void emplace_ast(Ts &&...ts) {
        ast_nodes.push(std::make_unique<AST>(std::forward<Ts>(ts)...));
    }

    static inline const char opchars[] = "+-*/=(,)";

    bool parse_atom() {  // atom := symbol | literial
        if (token->type == Token::Type::op)
            return false;
        emplace_ast(*token);
        token++;
        return true;
    }

    bool parse_funcall() {  // funcall := symbol "(" [expr ["," expr]*]? ")"
        if (token->type == Token::Type::reg) {
            token++;
            if (token->is_op({"("})) {
                auto opToken = *token++;
                if (token->is_op({")"})) {
                    emplace_ast(opToken);
                    return true;
                } else if (parse_expr()) {
                    std::vector<std::unique_ptr<AST>> arglist;
                    arglist.push_back(pop_ast());
                    while (token->is_op({","})) {
                        token++;
                        if (parse_expr()) {
                            arglist.push_back(pop_ast());
                        } else {
                            token--;
                            break;
                        }
                    }
                    if (token->is_op({")"})) {
                        token++;
                    }
                    emplace_ast(opToken, std::move(arglist));
                    return true;
                }
                token--;
            }
            token--;
        }
        return false;
    }

    bool parse_factor() {
        // factor := atom | <"+"|"-"> factor | "(" expr ")" | funcall
        if (parse_atom()) {
            return true;
        }
        if (token->is_op({"+", "-"})) {
            auto opToken = *token++;
            if (!parse_factor()) {
                token--;
                return false;
            }
            emplace_ast(opToken, pop_ast());
            return true;
        }
        if (token->is_op({"("})) {
            token++;
            if (!parse_expr()) {
                token--;
            }
            if (token->is_op({")"})) {
                token++;
            }
            return true;
        }
        return parse_funcall();
    }

    bool parse_term() {  // term := factor [<"*"|"/"|"%"> factor]*
        if (!parse_factor())
            return false;
        while (token->is_op({"*", "/", "%"})) {
            auto opToken = *token++;
            if (!parse_factor()) {
                token--;
                break;
            }
            emplace_ast(opToken, pop_ast(), pop_ast());
        }
        return true;
    }

    bool parse_expr() {  // expr := term [<"+"|"-"> term]*
        if (!parse_term())
            return false;
        while (token->is_op({"+", "-"})) {
            auto opToken = *token++;
            if (!parse_term()) {
                token--;
                break;
            }
            emplace_ast(opToken, pop_ast(), pop_ast());
        }
        return true;
    }

    bool parse_stmt() {  // stmt := expr ["=" expr]?
        if (!parse_expr())
            return false;
        if (token->is_op({"="})) {
            auto opToken = *token++;
            if (!parse_expr()) {
                token--;
            }
        }
        return true;
    }

    bool tokenize() {
        char head = 0;
        if (!(sin >> head))
            return false;
        if (isdigit(head)) {
            sin.unget();
            std::string ident;
            sin >> ident;
            tokens.emplace_back(Token::Type::imm, ident);
            return true;
        } else if (head == '@') {
            std::string ident;
            sin >> ident;
            tokens.emplace_back(Token::Type::mem, ident);
            return true;
        } else if (isalpha(head)) {
            sin.unget();
            std::string ident;
            sin >> ident;
            tokens.emplace_back(Token::Type::reg, ident);
            return true;
        }
        if (strchr(opchars, head)) {
            std::string op(1, head);
            tokens.emplace_back(Token::Type::op, op);
            return true;
        }
        sin.unget();
        return false;
    }
};

int main(void) {
    Parser p("posz = fit ( posx + 1, posy, posz )");
    while (p.tokenize());
    p.init_parse();
    p.parse_stmt();
    auto a = p.pop_ast();
    a->print();
    return 0;
}
