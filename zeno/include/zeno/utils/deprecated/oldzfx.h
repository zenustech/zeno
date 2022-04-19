#pragma once

/*******************\

 ZFX - the wrangling language for ZENO:  {Z=f(x)}

 This is a header-only library. To use it in your zeno project, just:

    #define ZFX_IMPLEMENTATION
    #include <zeno/zfx.h>

\*******************/

#include <cstdio>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <cmath>

#ifdef ZFX_IMPLEMENTATION
#include <magic_enum.hpp>
#include <iostream>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <cctype>
#include <stack>
#include <map>
#include <set>
#endif

namespace zfx {

struct Context {
    float regtable[256];
    float *memtable[256];

    constexpr float memfetch(int index) const {
        return *memtable[index];
    }

    constexpr void memstore(int index, float value) {
        *memtable[index] = value;
    }
};

enum class Opcode : int {
    mov, add, sub, mul, div, neg,
    sin, cos, tan, atan, asin, acos,
    abs, floor, ceil, sqrt, exp, log,
    min, max, pow, atan2, mod, big, les
};

enum class OperandType : int {
    imm, reg, mem,
};

struct Operand {
    OperandType type;
    union {
        int index;
        float value;
    };

    constexpr float get(Context *ctx) const {
        switch (type) {
        case OperandType::imm:
            return value;
        case OperandType::reg:
            return ctx->regtable[index];
        case OperandType::mem:
            return ctx->memfetch(index);
        }
        return 0;
    }

    constexpr void set(Context *ctx, float x) const {
        switch (type) {
        case OperandType::imm:
            return;
        case OperandType::reg:
            ctx->regtable[index] = x;
            return;
        case OperandType::mem:
            ctx->memstore(index, x);
            return;
        }
    }
};

struct Instruction {
    Opcode opcode;
    Operand dst, lhs, rhs;

    constexpr void execute(Context *ctx) const {
        float x = lhs.get(ctx);
        float y = rhs.get(ctx);
        float z = 0;
        switch (opcode) {
        case Opcode::mov: z = x; break;
        case Opcode::add: z = x + y; break;
        case Opcode::sub: z = x - y; break;
        case Opcode::mul: z = x * y; break;
        case Opcode::div: z = x / y; break;
        case Opcode::neg: z = -x; break;
        case Opcode::sin: z = std::sin(x); break;
        case Opcode::cos: z = std::cos(x); break;
        case Opcode::tan: z = std::tan(x); break;
        case Opcode::asin: z = std::asin(x); break;
        case Opcode::acos: z = std::acos(x); break;
        case Opcode::atan: z = std::atan(x); break;
        case Opcode::sqrt: z = std::sqrt(x); break;
        case Opcode::abs: z = std::fabs(x); break;
        case Opcode::floor: z = std::floor(x); break;
        case Opcode::ceil: z = std::ceil(x); break;
        case Opcode::exp: z = std::exp(x); break;
        case Opcode::log: z = std::log(x); break;
        case Opcode::min: z = std::min(x, y); break;
        case Opcode::max: z = std::max(x, y); break;
        case Opcode::pow: z = std::pow(x, y); break;
        case Opcode::atan2: z = std::atan2(x, y); break;
        case Opcode::mod: z = std::fmod(x, y); break;
        case Opcode::big: z = x>y?1.0f:0.0f; break;
        case Opcode::les: z = x>y?0.0f:1.0f; break;
        }
        dst.set(ctx, z);
    }
};

struct Program {
    std::vector<Instruction> insts;
    std::vector<std::string> channels;
    std::vector<std::string> parameters;

    void execute(Context *ctx) const {
        for (auto const &inst: insts) {
            inst.execute(ctx);
        }
    }
};

static inline auto split_str(std::string const &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(s);
    while (std::getline(iss, token, delimiter))
        tokens.push_back(token);
    return tokens;
}

Program *compile_program(std::string const &code);

#ifdef ZFX_IMPLEMENTATION
struct Assembler {
    Operand assemble_operand(std::string const &ident) {
        Operand operand;
        switch (ident[0]) {
        case '#': operand.type = OperandType::imm; break;
        case '$': operand.type = OperandType::reg; break;
        case '@': operand.type = OperandType::mem; break;
        }
        std::stringstream ss(ident.substr(1));
        if (operand.type == OperandType::imm)
            ss >> operand.value;
        else
            ss >> operand.index;
        return operand;
    }

    void bind_channel(int memid, std::string const &attr) {
        if (prog.channels.size() < memid + 1)
            prog.channels.resize(memid + 1);
        prog.channels[memid] = attr;
    }

    void bind_parameter(int parid, std::string const &name) {
        if (prog.parameters.size() < parid + 1)
            prog.parameters.resize(parid + 1);
        prog.parameters[parid] = name;
    }

    std::optional<Instruction> assemble_inst(std::string const &line) {
        auto tokens = split_str(line, ' ');
        assert(tokens.size() > 1);
        if (tokens[0] == "bind") {
            assert(tokens.size() > 2);
            int memid = 0;
            std::stringstream(tokens[1]) >> memid;
            bind_channel(memid, tokens[2]);
            return std::nullopt;
        }
        if (tokens[0] == "parm") {
            assert(tokens.size() > 2);
            int parid = 0;
            std::stringstream(tokens[1]) >> parid;
            bind_parameter(parid, tokens[2]);
            return std::nullopt;
        }
        assert(tokens.size() > 1);
        Instruction inst;
        auto opcode = magic_enum::enum_cast<Opcode>(tokens[0]);
        if (!opcode.has_value()) {
            printf("ERROR: invalid opcode: %s\n", tokens[0].c_str());
            return std::nullopt;
        }
        inst.opcode = opcode.value();
        inst.dst = assemble_operand(tokens[1]);
        if (tokens.size() > 2)
            inst.lhs = assemble_operand(tokens[2]);
        if (tokens.size() > 3)
            inst.rhs = assemble_operand(tokens[3]);
        return inst;
    }

    Program prog;

    void assemble(std::string const &lines) {
        for (auto const &line: split_str(lines, '\n')) {
            if (line.size() == 0)
                continue;
            auto inst = assemble_inst(line);
            if (inst.has_value()) {
                prog.insts.push_back(inst.value());
            }
        }
    }
};

static Program assemble_program(std::string const &lines) {
    Assembler a;
    a.assemble(lines);
    return a.prog;
}

template <class ...Ts>
static void error(Ts &&...ts) {
    (std::cerr << "ERROR: " << ... << ts) << std::endl;
    exit(-1);
}

struct Token {
    enum class Type {
        op, mem, reg, imm, call, none,
    } type;
    std::string ident;

    Token(Type type, std::string const &ident) : type(type), ident(ident) {}

    bool is_op(std::set<std::string> const &list) {
        return type == Type::op && list.find(ident) != list.end();
    }

    bool is_ident(std::set<std::string> const &list) {
        return type == Type::reg && list.find(ident) != list.end();
    }
};

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

    std::string dump() const {
        std::string res;
        if (args.size() != 0)
            res += "(";
        res += token.ident;
        for (auto const &arg: args) {
            res += " ";
            res += arg->dump();
        }
        if (args.size() != 0)
            res += ")";
        return res;
    }
};

class Parser {
private:
    std::string code;
    const char *cp;

    void skipws() {
        for (; *cp && isspace(*cp); cp++);
    }

public:
    Parser(std::string const &code_) : code(code_), cp(code.c_str()) {
        while (tokenize());
        init_parse();
    }

    auto parse() {
        while (parse_definition());
        std::vector<std::unique_ptr<AST>> asts;
        while (parse_stmt()) {
            asts.push_back(pop_ast());
        }
        return asts;
    }

    auto const &get_typing() const {
        return typing;
    }

    auto const &get_parnames() const {
        return parnames;
    }

private:
    std::vector<Token> tokens;
    decltype(tokens.begin()) token;

    void init_parse() {
        tokens.emplace_back(Token::Type::none, "EOF");
        token = tokens.begin();
    }

    std::map<std::string, std::string> typing;
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

    static inline const char opchars[] = "+-*/%=(,)";
    static inline const char *opstrs[] = {
        "+", "-", "*", "/", "%", "=", "(", ",", ")",
        "+=", "-=", "*=", "/=", "%=",
        NULL,
    };

    bool parse_atom() {  // atom := symbol | literial
        if (token->type == Token::Type::op)
            return false;
        if (token->type == Token::Type::none)
            return false;
        emplace_ast(*token);
        token++;
        return true;
    }

    bool parse_funcall() {  // funcall := symbol "(" [expr ["," expr]*]? ")"
        if (token->type == Token::Type::reg) {
            auto opToken = *token++;
            opToken.type = Token::Type::call;
            if (token->is_op({"("})) {
                token++;
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
        // factor := funcall | atom | <"+"|"-"> factor | "(" expr ")"
        if (parse_funcall()) {
            return true;
        }
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
        return false;
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

    std::vector<std::string> parnames;

    bool parse_definition() {
        // definition := "define" type symbol | "parname" symbol
        if (token->is_ident({"define"})) {
            auto opToken = *token++;
            auto type = *token++;
            if (type.type == Token::Type::reg) {
                auto symbol = *token++;
                if (symbol.type == Token::Type::mem) {
                    typing['@' + symbol.ident] = type.ident;
                    return true;
                } else if (symbol.type == Token::Type::reg) {
                    typing['$' + symbol.ident] = type.ident;
                    return true;
                }
                token--;
            }
            token--;
            token--;

        } else if (token->is_ident({"parname"})) {
            auto opToken = *token++;
            auto symbol = *token++;
            if (symbol.type == Token::Type::reg) {
                parnames.push_back(symbol.ident);
                return true;
            }
            token--;
            token--;
        }
        return false;
    }

    bool parse_stmt() {  // stmt := atom <"="|"+="|"-="|"*="|"/="|"%="> expr
        if (token->type == Token::Type::op)
            return false;
        if (token->type == Token::Type::none)
            return false;
        emplace_ast(*token);
        token++;
        if (!token->is_op({"=", "+=", "-=", "*=", "/=", "%="})) {
            token--;
            return false;
        }
        auto opToken = *token++;
        if (!parse_expr()) {
            token--;
            return false;
        }
        emplace_ast(opToken, pop_ast(), pop_ast());
        return true;
    }

    bool tokenize() {
        skipws();
        char head = *cp++;
        if (!head)
            return false;
        if (isdigit(head)) {
            std::string ident;
            ident += head;
            for (; isdigit(*cp) || *cp == '.'; ident += *cp++);
            tokens.emplace_back(Token::Type::imm, ident);
            return true;

        } else if (head == '@') {
            std::string ident;
            for (; isalnum(*cp) || strchr("_.:", *cp); ident += *cp++);
            tokens.emplace_back(Token::Type::mem, ident);
            return true;

        } else if (isalpha(head) || head == '_') {
            std::string ident;
            ident += head;
            for (; isalnum(*cp) || *cp == '.' || *cp == '_'; ident += *cp++);
            tokens.emplace_back(Token::Type::reg, ident);
            return true;
        }
        if (strchr(opchars, head)) {
            std::string op;
            op += head;
            while (*cp && strchr(opchars, *cp)) {
                const char **p;
                auto nop = op + *cp;
                for (p = opstrs; *p; p++) {
                    std::string s(*p);
                    if (s.substr(0, nop.size()) == nop)
                        break;
                }
                if (!*p)
                    break;
                op += *cp++;
            }
            tokens.emplace_back(Token::Type::op, op);
            return true;
        }
        cp--;
        return false;
    }
};

struct Transcriptor {
    struct Visit {
        std::string lvalue;
        std::string rvalue;
    };

    int tmpid = 0;

    std::string alloc_register() {
        char buf[233];
        sprintf(buf, "$_Tm%d", tmpid++);
        return buf;
    }

    std::map<std::string, std::string> regalloc;

    std::string get_register(std::string const &name) {
        auto it = regalloc.find(name);
        if (it == regalloc.end()) {
            auto reg = "$" + name;
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
            if (vis.rvalue.size() == 0) {
                error("invalid value encountered\n");
            }
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

    void inplace_movalue(char op, Visit &src, std::string const &dst) {
        emit(std::string() + op + " " + lvalue(src) + " " + dst + " " + dst);
    }

    Visit make_visit(std::string const &lvalue, std::string const &rvalue) {
        return {lvalue, rvalue};
    }

    Visit visit(AST *ast) {
        if (ast->token.type == Token::Type::op) {
            if (ast->token.is_op({"=", "+=", "-=", "*=", "/=", "%="})) {
                auto src = visit(ast->args[1].get());
                auto dst = visit(ast->args[0].get());
                if (ast->token.ident == "=") {
                    movalue(src, dst.lvalue);
                } else {
                    inplace_movalue(ast->token.ident[0], src, dst.lvalue);
                }
                return make_visit("", "");
            }
            auto res = ast->token.ident;
            for (auto const &arg: ast->args) {
                auto vis = visit(arg.get());
                res += " " + lvalue(vis);
            }
            return make_visit("", res);
        } else if (ast->token.type == Token::Type::call) {
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

static std::string tag_dim(std::string const &exp, int d) {
    if (exp[0] == '#')
        return exp;
    if (strchr(exp.c_str(), '.')) {
        return exp;
    }
    char buf[233];
    sprintf(buf, "%s.%d", exp.c_str(), d);
    return buf;
}

static std::string opchar_to_name(std::string const &op) {
    if (op == "+") return "add";
    if (op == "-") return "sub";
    if (op == "*") return "mul";
    if (op == "/") return "div";
    if (op == "%") return "mod";
    if (op == "=") return "mov";
    return op;
}

static std::string promote_type(std::string const &lhs,
    std::string const &rhs) {
    if (lhs[0] != rhs[0]) {
        error("cannot implicit promote type: ", lhs[0], " <=> ", rhs[0]);
    }
    char stype = lhs[0];
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

static int get_digit(char c) {
    return c <= '9' ? c - '0' : c - 'A';
}

static char put_digit(int n) {
    return n <= 9 ? n + '0' : n - 10 + 'A';
}

struct InitialPass {
    std::stringstream oss;

    void parse(std::string const &lines) {
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
            auto opinst = opchar_to_name(opcode);
            if (opcode == "-" && args.size() == 1) {
                opinst = "neg";
            }
            oss << opinst << " " << dst;
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

struct TypeCheck {
    std::map<std::string, std::string> typing;
    std::stringstream oss;

    std::string dump_typing() {
        oss.clear();
        for (auto const &[sym, type]: typing) {
            oss << "define " << type << " " << sym << '\n';
        }
        return oss.str();
    }

    void define_type(std::string const &exp, std::string const &typ) {
        typing[exp] = typ;
    }

    std::string determine_type(std::string const &exp) const {
        if (exp[0] == '#') {
            return "f1";
        }
        auto exps = split_str(exp, '.');
        if (exps.size() == 2) {
            auto it = typing.find(exps[0]);
            if (it == typing.end()) {
                error("cannot determine type of ", exp);
            }
            return it->second;
        }

        auto it = typing.find(exp);
        if (it == typing.end()) {
            error("cannot determine type of ", exp);
        }
        return it->second;
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
            for (int i = 2; i < ops.size(); i++) {
                auto arg = ops[i];
                auto type = determine_type(arg);
                argtypes.push_back(type);
            }
            auto dst = ops[1];
            op_promote_type(dst, opcode, argtypes);
        }
    }
};

struct UnfuncPass : TypeCheck {
    std::stringstream oss;

    int tmpid = 0;
    std::string alloc_register() {
        char buf[233];
        sprintf(buf, "$_Tp%d", tmpid++);
        return buf;
    }

    void emit_op(std::string const &opcode, std::string const &dst,
        std::vector<std::string> const &args) {

        if (opcode == "mla") {
            if (args.size() != 3) error("mla takes exactly 3 arguments\n");
            auto tmp = alloc_register();
            emit_op("mul", tmp, {args[0], args[1]});
            emit_op("add", dst, {tmp, args[2]});
            return;

        } else if (opcode == "mls") {
            if (args.size() != 3) error("mls takes exactly 3 arguments\n");
            auto tmp = alloc_register();
            emit_op("mul", tmp, {args[0], args[1]});
            emit_op("sub", dst, {tmp, args[2]});
            return;

        } else if (opcode == "inversesqrt") {
            emit_op("sqrt", dst, args);
            emit_op("div", dst, {"#1", dst});
            return;

        } else if (opcode == "clamp") {  // clamp(x, a, b) = min(max(x, a), b)
            if (args.size() != 3) error("clamp takes exactly 3 arguments\n");
            auto tmp = alloc_register();
            emit_op("max", tmp, {args[0], args[1]});
            emit_op("min", dst, {tmp, args[2]});
            return;

        } else if (opcode == "mix") {  // mix(x, y, a) = x + (y - x) * a
            if (args.size() != 3) error("mix takes exactly 3 arguments\n");
            auto tmp = alloc_register();
            emit_op("sub", tmp, {args[1], args[0]});
            emit_op("mul", tmp, {tmp, args[2]});
            emit_op("add", dst, {tmp, args[0]});
            return;

        } else if (opcode == "dot") {
            if (args.size() != 2) error("dot takes exactly 2 arguments\n");
            auto lhs = args[0], rhs = args[1];
            auto dim = get_digit(determine_type(lhs)[1]);
            auto rdim = get_digit(determine_type(rhs)[1]);
            if (dim != rdim) {
                error("vector dimension mismatch for dot: ", dim, " ", rdim);
            }
            auto tmp = alloc_register();
            emit_op("mul", dim == 1 ? dst : tmp,
                {tag_dim(lhs, 0), tag_dim(rhs, 0)});
            for (int d = 1; d < dim; d++) {
                emit_op("mla", d == dim - 1 ? dst : tmp,
                    {tag_dim(lhs, d), tag_dim(rhs, d), tmp});
            }
            return;

        } else if (opcode.substr(0, 3) == "vec") {
            int dim;
            std::istringstream(opcode.substr(3)) >> dim;
            if (args.size() != dim)
                error(opcode, " takes exactly ", dim, " arguments\n");

            auto tmp = alloc_register();
            for (int d = 0; d < dim; d++) {
                emit_op("mov", tag_dim(tmp, d), {args[d]});
            }
            std::ostringstream typss;
            typss << "f" << dim;
            auto typ = typss.str();
            define_type(tmp, typ);
            emit_op("mov", dst, {tmp});
            return;

        } else if (opcode == "cross") {
            if (args.size() != 2) error("cross takes exactly 2 arguments\n");
            auto lhs = args[0], rhs = args[1];
            auto dim = get_digit(determine_type(lhs)[1]);
            auto rdim = get_digit(determine_type(rhs)[1]);
            if (dim != rdim) {
                error("vector dimension mismatch for cross: ", dim, " ", rdim);
            }
            if (dim != 3) {
                error("cross only support 3d vectors for now");
            }
            auto tmp = alloc_register();
            for (int d = 0; d < 3; d++) {
                int e = (d + 1) % 3, f = (d + 2) % 3;
                emit_op("mul", tag_dim(tmp, d),
                    {tag_dim(lhs, e), tag_dim(rhs, f)});
                emit_op("mls", tag_dim(tmp, d),
                    {tag_dim(lhs, f), tag_dim(rhs, e), tag_dim(tmp, d)});
            }
            define_type(tmp, "f3");
            emit_op("mov", dst, {tmp});
            return;

        } else if (opcode == "length") {  // length(x) = sqrt(dot(x, x))
            if (args.size() != 1) error("length takes exactly 1 argument\n");
            auto src = args[0];
            auto tmp = alloc_register();
            emit_op("dot", tmp, {src, src});
            emit_op("sqrt", dst, {tmp});
            return;

        } else if (opcode == "normalize") {  // normalize(x) = x / length(x)
            if (args.size() != 1) error("normalize takes exactly 1 argument\n");
            auto src = args[0];
            auto tmp = alloc_register();
            emit_op("length", tmp, {src});
            emit_op("div", dst, {src, tmp});
            return;

        } else if (opcode == "distance") {  // distance(x, y) = length(y - x)
            if (args.size() != 2) error("distance takes exactly 2 arguments\n");
            auto tmp = alloc_register();
            emit_op("sub", tmp, {args[1], args[0]});
            emit_op("length", dst, {tmp});
            return;
        }
        // } else if (opcode == "big")
        // {
        //     if (args.size() != 2) error("big takes exactly 2 arguments\n");
        //     auto tmp = alloc_register();
        //     emit_op("big", tmp, {args[1], args[0]});
        //     return;
        // } else if (opcode == "les")
        // {
        //     if (args.size() != 2) error("les takes exactly 2 arguments\n");
        //     auto tmp = alloc_register();
        //     emit_op("les", tmp, {args[1], args[0]});
        //     return;
        // }

        oss << opcode << " " << dst;
        for (int i = 0; i < args.size(); i++) {
            oss << " " << args[i];
        }
        oss << '\n';
    }

    void parse(std::string const &lines) {
        type_check(lines);
        for (auto const &line: split_str(lines, '\n')) {
            if (line.size() == 0) return;
            auto ops = split_str(line, ' ');
            auto opcode = ops[0];
            std::vector<std::string> args;
            for (int i = 2; i < ops.size(); i++) {
                auto arg = ops[i];
                args.push_back(arg);
            }
            auto dst = ops[1];
            emit_op(opcode, dst, args);
        }
    }

    std::string dump() const {
        return oss.str();
    }
};

struct UnwrapPass : TypeCheck {
    std::stringstream oss;

    void emit_op(std::string const &opcode, std::string const &dst,
        std::vector<std::string> const &args) {
        auto dsttype = determine_type(dst);
        int dim = get_digit(dsttype[1]);
        for (int d = 0; d < dim; d++) {
            auto opinst = opcode;
            oss << opinst << " " << tag_dim(dst, d);
            for (auto const &arg: args) {
                auto argdim = get_digit(determine_type(arg)[1]);
                oss << " " << tag_dim(arg, d % argdim);
            }
            oss << '\n';
        }
    }

    void parse(std::string const &lines) {
        type_check(lines);
        for (auto const &line: split_str(lines, '\n')) {
            if (line.size() == 0) return;
            auto ops = split_str(line, ' ');
            auto opcode = ops[0];
            std::vector<std::string> args;
            for (int i = 2; i < ops.size(); i++) {
                auto arg = ops[i];
                args.push_back(arg);
            }
            auto dst = ops[1];
            emit_op(opcode, dst, args);
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
    std::map<std::string, int> parnames;

    void set_parnames(std::vector<std::string> const &pars) {
        for (auto const &par: pars) {
            auto id = regid++;
            parnames[par] = id;
            std::ostringstream idss; idss << '$' << id;
            assignment['$' + par] = idss.str();
        }
    }

    auto const &get_memories() const {
        return memories;
    }

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
            for (int i = 1; i < ops.size(); i++) {
                oss << " " << reassign_register(ops[i]);
            }
            oss << '\n';
        }
    }

    std::string dump() const {
        std::stringstream os;
        for (auto const &[key, id]: memories) {
            os << "bind " << id << " " << key << std::endl;
        }
        for (auto const &[key, id]: parnames) {
            os << "parm " << id << " " << key << std::endl;
        }
        os << oss.str();
        return os.str();
    }
};


#define ZFX_PRINT_IR
static std::string source_to_assembly(std::string const &code) {
#ifdef ZFX_PRINT_IR
    std::cout << "=== ZFX" << std::endl;
    std::cout << code << std::endl;
    std::cout << "=== Parser" << std::endl;
#endif

    Parser p(code + '\n');
    auto asts = p.parse();
#ifdef ZFX_PRINT_IR
    for (auto const &ast: asts) {
        std::cout << ast->dump() << std::endl;
    }
    std::cout << "=== Transcriptor" << std::endl;
#endif

    Transcriptor ts;
    for (auto const &ast: asts) {
        ts.visit(ast.get());
    }
    auto tsir = ts.dump();
#ifdef ZFX_PRINT_IR
    std::cout << tsir;
    std::cout << "=== InitialPass" << std::endl;
#endif

    InitialPass inp;
    inp.parse(tsir);
    auto inir = inp.dump();
#ifdef ZFX_PRINT_IR
    std::cout << inir;
    std::cout << "=== UnfuncPass" << std::endl;
#endif

    UnfuncPass ufp;
    ufp.typing = p.get_typing();
    ufp.parse(inir);
    auto ufir = ufp.dump();
#ifdef ZFX_PRINT_IR
    std::cout << ufp.dump_typing();
    std::cout << ufir;
    std::cout << "=== UnwrapPass" << std::endl;
#endif

    UnwrapPass uwp;
    uwp.typing = ufp.typing;
    uwp.parse(ufir);
    auto uwir = uwp.dump();
#ifdef ZFX_PRINT_IR
    std::cout << uwp.dump_typing();
    std::cout << uwir;
    std::cout << "=== ReassignPass" << std::endl;
#endif

    ReassignPass rap;
    rap.set_parnames(p.get_parnames());
    rap.parse(uwir);
    auto rair = rap.dump();
#ifdef ZFX_PRINT_IR
    std::cout << rair;
    std::cout << "=== Assemble" << std::endl;
#endif

    return rair;
}

struct Compiler {
    std::map<std::string, std::unique_ptr<Program>> cache;
        
    Program *compile(std::string const &code) {
        auto it = cache.find(code);
        if (it != cache.end()) {
            return it->second.get();
        }
        auto prog = std::make_unique<Program>(
            assemble_program(source_to_assembly(code)));
        auto rawptr = prog.get();
        cache[code] = std::move(prog);
        return rawptr;
    }
};

static Compiler main_compiler;

Program *compile_program(std::string const &code) {
    return main_compiler.compile(code);
}
#endif  // ifdef ZFX_IMPLEMENTATION

}
