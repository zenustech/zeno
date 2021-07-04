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

    void push_mem(std::string const &ident) {
        printf("mem: %s\n", ident.c_str());
    }

    void push_reg(std::string const &ident) {
        printf("reg: %s\n", ident.c_str());
    }

    void push_op(std::string const &op) {
        printf("op: %s\n", op.c_str());
    }

    void push_imm(float value) {
        printf("imm: %f\n", value);
    }

    bool toke() {
        char head = 0;
        if (!(ss >> head))
            return false;
        if (isdigit(head)) {
            ss.unget();
            float value;
            ss >> value;
            push_imm(value);
            return true;
        } else if (head == '@' || isalpha(head)) {
            if (head != '@')
                ss.unget();
            std::string ident;
            ss >> ident;
            if (head == '@')
                push_mem(ident);
            else
                push_reg(ident);
            return true;
        }

        if (strchr("+-*/=", head)) {
            std::string op(1, head);
            push_op(op);
            return true;
        }
        ss.unget();
        return false;
    }
};

int main(void) {
    Parser p("@posz = 1");
    while (p.toke()) {
    }
    return 0;
}
