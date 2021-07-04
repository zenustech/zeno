#include "program.h"
#include "ast.h"
#include <sstream>
#include <iostream>
#include <cstdlib>

using std::cout;
using std::endl;

auto parse_program(std::string const &code) {
    Parser p(code);
    return p.parse();
}

std::string opchar_to_opcode(std::string const &op) {
    return "add";
}

struct Translator {
    struct Visit {
        std::string lvalue;
        std::string rvalue;
    };

    int regid = 0;

    std::string alloc_register() {
        char buf[233];
        sprintf(buf, "$%d", regid++ % 256);
        return buf;
    }

    void emit(std::string const &str) {
        printf("[%s]\n", str.c_str());
    }

    std::string lvalue(Visit &vis) {
        if (vis.lvalue.size() == 0) {
            auto reg = alloc_register();
            vis.lvalue = reg;
            emit(vis.rvalue + " " + reg);
        }
        return vis.lvalue;
    }

    Visit make_visit(std::string const &lvalue, std::string const &rvalue) {
        return {lvalue, rvalue};
    }

    Visit visit(AST *ast) {
        if (ast->token.type == Token::Type::op) {
            auto res = opchar_to_opcode(ast->token.ident);
            for (auto const &arg: ast->args) {
                auto vis = visit(arg.get());
                res += " " + lvalue(vis);
            }
            return make_visit("", res);
        } else if (ast->token.type == Token::Type::mem) {
            return make_visit("@" + ast->token.ident, "");
        } else if (ast->token.type == Token::Type::reg) {
            return make_visit("$" + ast->token.ident, "");
        } else if (ast->token.type == Token::Type::imm) {
            return make_visit("#" + ast->token.ident, "");
        }
        return make_visit("", "");
    }

    std::string lines;

    std::string get_assembly() {
        return lines;
    }
};

std::string translate_program(AST *ast) {
    Translator t;
    t.visit(ast);
    return t.get_assembly();
}

int main() {
    Parser par("2 + 3 + 1");
    auto ast = par.parse();
    ast->print();
    translate_program(ast.get());
    return 0;
}
