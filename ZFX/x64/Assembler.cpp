#include "x64/SIMDBuilder.h"
#include "x64/Executable.h"
#include "x64/Program.h"
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
                auto offset = from_string<int>(linesep[2]);
                offset *= SIMDBuilder::scalarSizeOfType(simdkind);
                builder->addAvxBroadcastLoadOp(simdkind,
                    dst, {opreg::rcx, memflag::reg_imm8, offset});

            } else if (cmd == "ldl") {  // rbx points to an array of variables
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto offset = from_string<int>(linesep[2]);
                offset *= SIMDBuilder::sizeOfType(simdkind);
                builder->addAvxMemoryOp(simdkind, opcode::loadu,
                    dst, {opreg::rbx, memflag::reg_imm8, offset});

            } else if (cmd == "stl") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto offset = from_string<int>(linesep[2]);
                offset *= SIMDBuilder::sizeOfType(simdkind);
                builder->addAvxMemoryOp(simdkind, opcode::storeu,
                    dst, {opreg::rcx, memflag::reg_imm8, offset});

            } else if (cmd == "ldg") {  // rdx points to an array of pointers
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto offset = from_string<int>(linesep[2]);
                offset *= sizeof(void *);
                builder->addRegularLoadOp(opreg::rax,
                    {opreg::rdx, memflag::reg_imm8, offset});
                builder->addAvxMemoryOp(simdkind, opcode::loadu,
                    dst, opreg::rax);

            } else if (cmd == "stg") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto offset = from_string<int>(linesep[2]);
                offset *= sizeof(void *);
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
};

std::unique_ptr<Program> assemble_program(std::string const &lines) {
    Assembler a;
    a.parse(lines);
    a.builder->addReturn();

    auto const &insts = a.builder->getResult();
    for (auto const &inst: insts) printf("%02X ", inst); printf("\n");

    auto prog = std::make_unique<Program>();
    prog->executable = std::make_unique<ExecutableInstance>(insts);

    return prog;
};
