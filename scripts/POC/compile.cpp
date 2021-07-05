#include "program.h"
#include "ast.h"
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <map>

using std::cout;
using std::cerr;
using std::endl;

struct Transcriptor {
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
        lines += str + "\n";
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

static std::string opchar_to_name(std::string const &op) {
    if (op == "+") return "add";
    if (op == "-") return "sub";
    if (op == "*") return "mul";
    if (op == "/") return "div";
    if (op == "mov") return "mov";
    return "";
}

static int get_digit(char c) {
    return c <= '9' ? c - '0' : c - 'A';
}

static char put_digit(int n) {
    return n <= 9 ? n + '0' : n - 10 + 'A';
}

template <class ...Ts>
static void error(Ts &&...ts) {
    (cerr << "ERROR: " << ... << ts) << endl;
    exit(-1);
}

struct UnwrapPass {
    std::map<std::string, std::string> typing;
    std::stringstream oss;

    UnwrapPass() {
        typing["@a"] = "f3";
        typing["@b"] = "f1";
    }

    std::string determine_type(std::string const &exp) const {
        if (exp[0] == '#') {
            return strchr(exp.substr(1).c_str(), '.') ? "f1" : "i1";
        }
        auto it = typing.find(exp);
        if (it == typing.end()) {
            error("cannot determine type of ", exp);
        }
        return it->second;
    }

    std::string tag_dim(std::string const &exp, int d) const {
        if (exp[0] == '#')
            return exp;
        char buf[233];
        sprintf(buf, "%s.%d", exp.c_str(), d);
        return buf;
    }

    void emit_op(std::string const &opcode, std::string const &dst,
        std::vector<std::string> const &args) {
        auto dsttype = determine_type(dst);
        int dim = get_digit(dsttype[1]);
        for (int d = 0; d < dim; d++) {
            auto opinst = dsttype[0] + opchar_to_name(opcode);
            oss << opinst << " " << tag_dim(dst, d);
            for (auto const &arg: args) {
                auto argdim = get_digit(determine_type(arg)[1]);
                oss << " " << tag_dim(arg, d % argdim);
            }
            oss << '\n';
        }
    }

    static std::string promote_type(std::string const &lhs,
        std::string const &rhs) {
        char stype = lhs[0] <= rhs[0] ? lhs[0] : rhs[0];
        char dim = 0;
        if (lhs[1] == '1') {
            dim = rhs[1];
        } else if (rhs[1] == '1') {
            dim = lhs[1];
        } else {
            if (lhs[1] != rhs[1]) {
                error("vector dimension mismatch: ", lhs[1], " != ", rhs[1]);
            }
            dim = lhs[1];
        }
        return std::string() + stype + dim;
    }

    void op_promote_type(std::string const &dst,
        std::string const &opcode, std::vector<std::string> const &types) {
        auto curtype = types[0];
        for (int i = 1; i < types.size(); i++) {
            auto const &type = types[i];
            curtype = promote_type(curtype, type);
        }
        auto it = typing.find(dst);
        if (it == typing.end()) {
            typing[dst] = curtype;
        } else {
            if (it->second != curtype) {
                if (dst[0] == '@') {
                    if (promote_type(it->second, curtype) != it->second) {
                        error("cannot cast: ", it->second, " <- ", curtype);
                    }
                }
                typing[dst] = promote_type(it->second, curtype);
            }
        }
    }

    void type_check(std::string const &lines) {
        for (auto const &line: split_str(lines, '\n')) {
            if (line.size() == 0) return;
            auto ops = split_str(line, ' ');
            auto opcode = ops[0];
            std::vector<std::string> argtypes;
            for (int i = 1; i < ops.size() - 1; i++) {
                auto arg = ops[i];
                auto type = determine_type(arg);
                argtypes.push_back(type);
            }
            auto dst = ops[ops.size() - 1];
            op_promote_type(dst, opcode, argtypes);
        }
    }

    void emission(std::string const &lines) {
        for (auto const &line: split_str(lines, '\n')) {
            if (line.size() == 0) return;
            auto ops = split_str(line, ' ');
            auto opcode = ops[0];
            std::vector<std::string> args;
            for (int i = 1; i < ops.size() - 1; i++) {
                auto arg = ops[i];
                args.push_back(arg);
            }
            auto dst = ops[ops.size() - 1];
            emit_op(opcode, dst, args);
        }
    }

    void parse(std::string const &lines) {
        type_check(lines);
        emission(lines);
    }

    std::string dump() const {
        return oss.str();
    }

    auto const &get_typing() const {
        return typing;
    }
};

struct UntypePass {
    std::stringstream oss;
    std::map<std::string, char> typing;

    void set_typing(std::map<std::string, std::string> const &uwTyping) {
        for (auto const &[exp, type]: uwTyping) {
            auto dim = get_digit(type[1]);
            auto ptype = type[0];
            for (int d = 0; d < dim; d++) {
                char buf[233];
                sprintf(buf, "%s.%d", exp.c_str(), d);
                //printf("deftype %s %c\n", buf, ptype);
                typing[buf] = ptype;
            }
        }
    }

    char determine_type(std::string const &exp) const {
        if (exp[0] == '#') {
            return strchr(exp.substr(1).c_str(), '.') ? 'f' : 'i';
        }
        auto it = typing.find(exp);
        if (it == typing.end()) {
            error("cannot determine type of ", exp);
        }
        return it->second;
    }

    int tmpid = 0;
    std::string emit_cast(char src, char dst, std::string const &exp) {
        char tmp[233];
        sprintf(tmp, "$tmp%d", tmpid++);
        oss << "cvt" << dst << src << " " << tmp << " " << exp << '\n';
        return tmp;
    }

    void parse(std::string const &lines) {
        for (auto const &line: split_str(lines, '\n')) {
            if (line.size() == 0) return;
            auto ops = split_str(line, ' ');
            auto opcode = ops[0];
            auto dst = ops[1];
            std::vector<std::string> args;
            for (int i = 2; i < ops.size(); i++) {
                auto arg = ops[i];
                args.push_back(arg);
            }
            auto dsttype = determine_type(dst);
            for (int i = 0; i < args.size(); i++) {
                auto argtype = determine_type(args[i]);
                if (argtype != dsttype && args[i][0] != '#') {
                    args[i] = emit_cast(argtype, dsttype, args[i]);
                }
            }
            oss << opcode << " " << dst;
            for (int i = 0; i < args.size(); i++) {
                oss << " " << args[i];
            }
            oss << '\n';
        }
    }

    std::string dump() const {
        return oss.str();
    }
};

struct ReassignPass {
    std::stringstream oss;

    int regid = 0;
    int memid = 0;
    std::map<std::string, std::string> assignment;
    std::map<std::string, int> memories;

    std::string reassign_register(std::string const &exp) {
        if (exp[0] == '@') {
            auto key = exp.substr(1);
            auto it = memories.find(key);
            int id;
            if (it != memories.end()) {
                id = it->second;
            } else {
                id = memid++;
                memories[key] = id;
            }
            char buf[233];
            sprintf(buf, "@%d", id);
            return buf;
        }

        if (exp[0] != '$')
            return exp;
        auto it = assignment.find(exp);
        if (it != assignment.end()) {
            return it->second;
        }
        char buf[233];
        sprintf(buf, "$%d", regid++);
        assignment[exp] = buf;
        return buf;
    }

    void parse(std::string const &lines) {
        for (auto const &line: split_str(lines, '\n')) {
            if (line.size() == 0) return;
            auto ops = split_str(line, ' ');
            oss << ops[0];
            for (int i = 2; i < ops.size(); i++) {
                oss << " " << reassign_register(ops[i]);
            }
            oss << '\n';
        }
        /*for (auto const &[key, id]: memories) {
            oss << "def @" << id << " " << key << endl;
        }*/
    }

    std::string dump() const {
        return oss.str();
    }
};

int main() {
    auto code = "@a = @a + @b * (3 + 1.4)";

    cout << "===" << endl;
    cout << code << endl;
    cout << "===" << endl;

    Parser p(code);
    auto ast = p.parse();
    cout << ast->dump() << endl;
    cout << "===" << endl;

    Transcriptor t;
    t.visit(ast.get());
    ast = nullptr;
    auto ir = t.dump();
    cout << ir;
    cout << "===" << endl;

    UnwrapPass uwp;
    uwp.parse(ir);
    auto uwir = uwp.dump();
    cout << uwir;
    cout << "===" << endl;

    UntypePass utp;
    utp.set_typing(uwp.get_typing());
    utp.parse(uwir);
    auto utir = utp.dump();
    cout << utir;
    cout << "===" << endl;

    ReassignPass rap;
    rap.parse(uwir);
    auto rair = rap.dump();
    cout << rair;
    cout << "===" << endl;

    return 0;
}
