#include "assembler/SIMDBuilder.h"
#include "assembler/Executable.h"
#include "assembler/Program.h"
#include "common.h"
#include <sstream>
#include <map>

#define ERROR_IF(x) do { \
    if (x) { \
        error("Assertion `%s` failed", #x); \
    } \
} while (0)

struct Assembler {
    int simdkind = optype::xmmps;
    std::unique_ptr<SIMDBuilder> builder = std::make_unique<SIMDBuilder>();

    std::map<std::string, int> symtable;
    std::map<std::string, int> consttable;
    int constid = 0, symid = 0;

    int lookup_constant_offset(std::string const &expr) {
        auto it = consttable.find(expr);
        if (it != consttable.end())
            return it->second;
        auto id = constid++;
        consttable[expr] = id;
        return id;
    }

    int lookup_symbol_offset(std::string const &sym) {
        auto it = symtable.find(sym);
        if (it != symtable.end())
            return it->second;
        //error("undefined symbol `%s`", sym.c_str());
        auto id = symid++;
        symtable[sym] = id;
        return id;
    }

    void parse(std::string const &lines) {
        for (auto line: split_str(lines, '\n')) {
            if (!line.size()) continue;

            auto linesep = split_str(line, ' ');
            ERROR_IF(linesep.size() < 1);
            auto cmd = linesep[0];
            if (0) {

            } else if (cmd == "ldi") {  // rcx points to an array of constants
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto offset = lookup_constant_offset(
                    linesep[2]) * SIMDBuilder::scalarSizeOfType(simdkind);
                builder->addAvxBroadcastLoadOp(simdkind,
                    dst, {opreg::rcx, memflag::reg_imm8, offset});

            } else if (cmd == "lds") {  // rdx points to an array of pointers
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto offset = lookup_symbol_offset(
                    linesep[2]) * sizeof(void *);
                builder->addRegularLoadOp(opreg::rax,
                    {opreg::rdx, memflag::reg_imm8, offset});
                builder->addAvxMemoryOp(simdkind, opcode::loadu,
                    dst, opreg::rax);

            } else if (cmd == "sts") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto offset = lookup_symbol_offset(linesep[2]);
                builder->addRegularLoadOp(opreg::rax,
                    {opreg::rdx, memflag::reg_imm8, offset});
                builder->addAvxMemoryOp(simdkind, opcode::storeu,
                    dst, opreg::rax);

            } else if (cmd == "add") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::add,
                    dst, lhs, rhs);

            } else if (cmd == "sub") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::sub,
                    dst, lhs, rhs);

            } else if (cmd == "mul") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::mul,
                    dst, lhs, rhs);

            } else if (cmd == "div") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::div,
                    dst, lhs, rhs);

            } else if (cmd == "mov") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto src = from_string<int>(linesep[2]);
                builder->addAvxMoveOp(dst, src);

            } else {
                error("bad assembly command `%s`", cmd.c_str());
            }
        }
    }

    void prepareConstants(std::vector<float> &consts) const {
        consts.resize(constid);
        for (auto const &[expr, id]: consttable) {
            float value = 0;
            if (!(std::stringstream(expr) >> value))
                error("failed to parse literial constant `%s`", expr.c_str());
            consts[id] = value;
        }
    }
};

std::unique_ptr<Program> assemble_program(std::string const &lines) {
    Assembler assembler;
    assembler.parse(lines);
    assembler.builder->addReturn();

    auto const &insts = assembler.builder->getResult();
    for (auto const &inst: insts) printf("%02X ", inst); printf("\n");

    auto prog = std::make_unique<Program>();
    prog->executable = std::make_unique<ExecutableInstance>(insts);
    prog->symtable = assembler.symtable;
    assembler.prepareConstants(prog->consts);

    return prog;
};
