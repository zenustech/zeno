#include <vector>
#include <type_traits>
#include <magic_enum.hpp>
#include <iostream>
#include <cassert>
#include <sstream>
#include <cstring>
#include <cctype>
#include <array>

struct Parser {
    std::stringstream ss;

    Parser(std::string const &code) : ss(code) {}

    struct Token {
        enum class Type {
            mem, reg, op, imm,
        } type;
        std::string ident;
        float value = 0;

        Token(Type type, std::string const &ident) : type(type), ident(ident) {}
        Token(Type type, float value) : type(type), value(value) {}
    };

    std::vector<Token> tokens;

    bool tokenize() {
        char head = 0;
        if (!(ss >> head))
            return false;
        if (isdigit(head)) {
            ss.unget();
            float value;
            ss >> value;
            tokens.emplace_back(Token::Type::imm, value);
            return true;
        } else if (head == '@') {
            std::string ident;
            ss >> ident;
            tokens.emplace_back(Token::Type::mem, ident);
            return true;
        } else if (isalpha(head)) {
            ss.unget();
            std::string ident;
            ss >> ident;
            tokens.emplace_back(Token::Type::reg, ident);
            return true;
        }
        if (strchr("+-*/=", head)) {
            std::string op(1, head);
            tokens.emplace_back(Token::Type::op, op);
            return true;
        }
        ss.unget();
        return false;
    }
};

int main(void) {
    Parser p("@posz = @posx + 1");
    while (p.tokenize());
    return 0;
}
