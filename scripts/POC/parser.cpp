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
            op, mem, reg, imm,
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
        token = tokens.begin();
    }

    struct AST {
        std::unique_ptr<AST> lhs, rhs;
        Token token;

        AST
            ( std::unique_ptr<AST> lhs
            , std::unique_ptr<AST> rhs
            , Token token
            )
            : lhs(std::move(lhs))
            , rhs(std::move(rhs))
            , token(std::move(token))
            {}
        explicit AST
            ( Token token
            )
            : lhs(nullptr)
            , rhs(nullptr)
            , token(std::move(token))
            {}
    };

    std::stack<std::unique_ptr<AST>> asts;

    bool parse_atom() {
        if (token->type == Token::Type::op)
            return false;
        asts.push(std::make_unique<AST>(*token));
        token++;
        return true;
    }

    bool parse_factor() {  // factor := [<"+"|"-">]? atom | "(" expr ")"
        return parse_atom();
    }

    bool parse_term() {  // term := factor [<"*"|"/"|"%"> factor]*
        if (!parse_factor())
            return false;
        while (token->is_op({"*", "/", "%"})) {
            token++;
            if (!parse_factor())
                break;
        }
        return true;
    }

    bool parse_expr() {  // expr := term [<"+"|"-"> term]*
        if (!parse_term())
            return false;
        while (token->is_op({"+", "-"})) {
            token++;
            if (!parse_term())
                break;
        }
        return true;
    }

    bool parse_stmt() {  // stmt := expr ["=" expr]?
        if (!parse_expr())
            return false;
        if (token->is_op({"="})) {
            token++;
            parse_expr();
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
        if (strchr("+-*/=", head)) {
            std::string op(1, head);
            tokens.emplace_back(Token::Type::op, op);
            return true;
        }
        sin.unget();
        return false;
    }
};

int main(void) {
    Parser p("@posz = @posx + @posy - 3.14");
    while (p.tokenize());
    p.init_parse();
    p.parse_stmt();
    return 0;
}
