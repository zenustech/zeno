#include <vector>
#include <type_traits>
#include <magic_enum.hpp>
#include <iostream>
#include <cassert>
#include <sstream>
#include <cstring>
#include <cctype>
#include <array>
#include <stack>
#include <set>

struct Parser {
    std::istringstream sin;
    std::ostringstream sout;

    Parser(std::string const &code) : sin(code) {}

    struct Token {
        enum class Type {
            op, mem, reg, imm,
        } type;
        std::string ident;
        float value = 0;

        Token(Type type, std::string const &ident) : type(type), ident(ident) {}
        Token(Type type, float value) : type(type), value(value) {}

        bool is_op(std::set<std::string> const &list) {
            return type == Type::op && list.find(ident) != list.end();
        }
    };

    std::vector<Token> tokens;
    decltype(tokens.begin()) token;

    void init_parse() {
        token = tokens.begin();
    }

    bool parse_atom() {
        if (token->type == Token::Type::op)
            return false;
        if (token->type == Token::Type::imm)
            printf("[%f]\n", token->value);
        else
            printf("[%s]\n", token->ident.c_str());
        token++;
        return true;
    }

    bool parse_factor() {  // factor := [<"+"|"-">]? atom | "(" expr ")"
        return parse_atom();
    }

    bool parse_term() {  // term := factor [<"*"|"/"|"%"> factor]*
        return parse_factor();
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
            float value;
            sin >> value;
            tokens.emplace_back(Token::Type::imm, value);
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
