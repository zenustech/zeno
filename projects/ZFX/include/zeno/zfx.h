#pragma once

#ifdef ZFX_IMPLEMENTATION
#include <magic_enum.hpp>
#include <iostream>
#include <cstring>
#include <cassert>
#include <sstream>
#include <cstdlib>
#include <string>
#include <memory>
#include <vector>
#include <cstdio>
#include <cctype>
#include <cmath>
#include <stack>
#include <map>
#include <set>
#else
#include <cstdio>
#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include <map>
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
    mov, add, sub, mul, div,
    sin, cos, tan, atan, asin, acos,
    abs, floor, ceil, sqrt, exp, log,
    min, max, pow, atan2, mod,
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
        }
        dst.set(ctx, z);
    }
};

struct Program {
    std::vector<Instruction> insts;
    std::vector<std::string> channels;

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

    std::optional<Instruction> assemble_inst(std::string const &line) {
        auto tokens = split_str(line, ' ');
        assert(tokens.size() > 1);
        if (tokens[0] == "bind") {
            assert(tokens.size() > 3);
            int memid = 0;
            std::stringstream(tokens[1]) >> memid;
            bind_channel(memid, tokens[2]);
            return std::nullopt;
        }
        assert(tokens.size() > 2);
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

    static inline const char opchars[] = "+-*/=(,)";

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

    bool parse_definition() {  // definition := "define" type symbol
        if (token->is_ident({"define"})) {
            auto opToken = *token++;
            auto type = *token++;
            if (type.type == Token::Type::reg) {
                auto symbol = *token++;
                if (symbol.type == Token::Type::mem) {
                    typing[symbol.ident] = type.ident;
                    return true;
                }
                token--;
            }
            token--;
            token--;
        }
        return false;
    }

    bool parse_stmt() {  // stmt := expr ["=" expr]?
        if (!parse_expr())
            return false;
        if (token->is_op({"="})) {
            auto opToken = *token++;
            if (!parse_expr()) {
                token--;
            }
            emplace_ast(opToken, pop_ast(), pop_ast());
        }
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
            for (; isalnum(*cp) || *cp == '.'; ident += *cp++);
            tokens.emplace_back(Token::Type::mem, ident);
            return true;

        } else if (isalpha(head)) {
            std::string ident;
            ident += head;
            for (; isalnum(*cp) || *cp == '.'; ident += *cp++);
            tokens.emplace_back(Token::Type::reg, ident);
            return true;
        }
        if (strchr(opchars, head)) {
            std::string op;
            op += head;
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

    void set_parser_typing(std::map<std::string, std::string> const &typ) {
        for (auto const &[symb, type]: typ) {
            typing['@' + symb] = type;
        }
    }

    std::string dump_typing() {
        oss.clear();
        for (auto const &[sym, type]: typing) {
            oss << "define " << type << " " << sym << '\n';
        }
        return oss.str();
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

    std::string determine_type(std::string const &exp) const {
        if (exp[0] == '#') {
            return "f1";
        }
        auto exps = split_str(exp, '.');
        if (exps.size() == 2) {
            auto it = typing.find(exps[0]);
            if (it == typing.end()) {
                error("cannot determine component type of ", exp);
            }
            return it->second;
        }

        auto it = typing.find(exp);
        if (it == typing.end()) {
            error("cannot determine type of ", exp);
        }
        return it->second;
    }

    int tmpid = 0;
    std::string alloc_register() {
        char buf[233];
        sprintf(buf, "$_Tp%d", tmpid++);
        return buf;
    }

    void emit_op(std::string const &opcode, std::string const &dst,
        std::vector<std::string> const &args) {

        if (opcode == "rsqrt") {
            auto tmp = alloc_register();
            emit_op("sqrt", tmp, args);
            emit_op("div", dst, {"#1", tmp});
            return;

        } else if (opcode == "dot") {
            if (args.size() != 2) error("dot takes exactly 2 arguments\n");
            auto lhs = args[0], rhs = args[1];
            auto dim = get_digit(determine_type(lhs)[1]);
            auto ldim = get_digit(determine_type(rhs)[1]);
            if (dim != ldim) {
                error("vector dimension mismatch for dot: ", dim, " ", ldim);
            }
            for (int d = 0; d < dim; d++) {
                auto tmp = alloc_register();
                emit_op("mul", d == 0 ? dst : tmp,
                    {tag_dim(lhs, d), tag_dim(rhs, d)});
                if (d != 0)
                    emit_op("add", dst, {dst, tmp});
            }
            return;

        } else if (opcode == "length") {
            if (args.size() != 1) error("length takes exactly 1 argument\n");
            auto src = args[0];
            emit_op("dot", dst, {src, src});
            emit_op("sqrt", dst, {dst});
            return;
        }

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

    Parser p(code);
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
    ufp.set_parser_typing(p.get_typing());
    ufp.parse(inir);
    auto ufir = ufp.dump();
#ifdef ZFX_PRINT_IR
    std::cout << ufir;
    std::cout << "=== UnwrapPass" << std::endl;
#endif

    UnwrapPass uwp;
    uwp.set_parser_typing(p.get_typing());
    uwp.parse(ufir);
    auto uwir = uwp.dump();
#ifdef ZFX_PRINT_IR
    std::cout << uwir;
    std::cout << "=== ReassignPass" << std::endl;
#endif

    ReassignPass rap;
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
#else  // ifdef ZFX_IMPLEMENTATION
Program *compile_program(std::string const &code);
#endif  // ifdef ZFX_IMPLEMENTATION

}
