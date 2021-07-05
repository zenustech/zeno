#include "program.h"
#include "ast.h"
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <map>

using std::cout;
using std::endl;

struct Translator {
    struct Visit {
        std::string lvalue;
        std::string rvalue;
    };

    int regid = 0;

    std::string alloc_register() {
        char buf[233];
        sprintf(buf, "$%d", regid++);
        return buf;
    }

    std::map<std::string, std::string> regalloc;

    std::string get_register(std::string const &name) {
        auto it = regalloc.find(name);
        if (it == regalloc.end()) {
            auto reg = alloc_register();
            regalloc[name] = reg;
            return reg;
        }
        return it->second;
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

    void movalue(Visit &src, std::string const &dst) {
        if (src.lvalue.size() == 0) {
            src.lvalue = dst;
            emit(src.rvalue + " " + dst);
        } else {
            emit("mov " + src.lvalue + " " + dst);
        }
    }

    Visit make_visit(std::string const &lvalue, std::string const &rvalue) {
        return {lvalue, rvalue};
    }

    Visit visit(AST *ast) {
        if (ast->token.type == Token::Type::op) {
            if (ast->token.ident == "=") {
                auto src = visit(ast->args[1].get());
                auto dst = visit(ast->args[0].get());
                movalue(src, dst.lvalue);
                return make_visit("", "");
            }
            auto res = ast->token.ident;
            for (auto const &arg: ast->args) {
                auto vis = visit(arg.get());
                res += " " + lvalue(vis);
            }
            return make_visit("", res);
        } else if (ast->token.type == Token::Type::mem) {
            return make_visit("@" + ast->token.ident, "");
        } else if (ast->token.type == Token::Type::reg) {
            return make_visit(get_register(ast->token.ident), "");
        } else if (ast->token.type == Token::Type::imm) {
            return make_visit("#" + ast->token.ident, "");
        }
        return make_visit("", "");
    }

    std::string lines;

    std::string dump() const {
        return lines;
    }
};

static std::vector<std::string> split_str(std::string const &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream iss(s);
  while (std::getline(iss, token, delimiter))
    tokens.push_back(token);
  return tokens;
}

struct Finalizer {
    void parse(std::string const &lines) {
        for (auto const &line: split_str(lines, '\n')) {
            auto ops = split_str(line, ' ');
            auto opcode = ops[0];
            std::vector<std::string> args;
            for (int i = 1; i < ops.size() - 1; i++) {
                args.push_back(ops[i]);
            }
            auto dst = ops[ops.size() - 1];
        }
    }

    std::string dump() const {
        return "done";
    }
};

auto parse_program(std::string const &code) {
    Parser p(code);
    return p.parse();
}

std::string translate_program(AST *ast) {
    Translator t;
    t.visit(ast);
    return t.dump();
}

std::string finalize_program(std::string const &ir) {
    Finalizer f;
    f.parse(ir);
    return f.dump();
}

int main() {
    auto stm = "@a = 4 * @a + 3 * @b";
    cout << stm << endl;
    Parser par(stm);
    auto ast = par.parse();
    auto ir = translate_program(ast.get());
    auto assem = finalize_program(ir);
    cout << assem << endl;
    return 0;
}
