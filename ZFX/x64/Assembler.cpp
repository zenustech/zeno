#include "x64/SIMDBuilder.h"
#include "x64/Executable.h"
#include "x64/Program.h"
#include "common.h"
#include <sstream>
#include <map>

namespace zfx::x64 {

#define ERROR_IF(x) do { \
    if (x) { \
        error("Assertion `%s` failed", #x); \
    } \
} while (0)

struct Assembler {
    int simdkind = optype::xmmps;
    std::unique_ptr<SIMDBuilder> builder = std::make_unique<SIMDBuilder>();
    std::unique_ptr<Program> prog = std::make_unique<Program>();

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
                auto value_expr = linesep[2];
                float value = from_string<float>(value_expr);
                int id = prog->consts.size();
                // yeah: we assumed simdkind to be xmmps in this branch
                // todo: use template programming to be generic on this
                int offset = id * sizeof(float);
                prog->consts.push_back(value);
                builder->addAvxBroadcastLoadOp(simdkind,
                    dst, {opreg::rcx, memflag::reg_imm8, offset});

            } else if (cmd == "ldl") {  // rbx points to an array of variables
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                if (prog->locals.size() < id + 1)
                    prog->locals.resize(id + 1);
                int offset = id * SIMDBuilder::sizeOfType(simdkind);
                builder->addAvxMemoryOp(simdkind, opcode::loadu,
                    dst, {opreg::rbx, memflag::reg_imm8, offset});

            } else if (cmd == "stl") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                if (prog->locals.size() < id + 1)
                    prog->locals.resize(id + 1);
                int offset = id * SIMDBuilder::sizeOfType(simdkind);
                builder->addAvxMemoryOp(simdkind, opcode::storeu,
                    dst, {opreg::rbx, memflag::reg_imm8, offset});

            } else if (cmd == "ldg") {  // rdx points to an array of pointers
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                if (prog->chptrs.size() < id + 1)
                    prog->chptrs.resize(id + 1);
                int offset = id * sizeof(void *);
                builder->addRegularLoadOp(opreg::rax,
                    {opreg::rdx, memflag::reg_imm8, offset});
                builder->addAvxMemoryOp(simdkind, opcode::loadu,
                    dst, opreg::rax);

            } else if (cmd == "stg") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                if (prog->chptrs.size() < id + 1)
                    prog->chptrs.resize(id + 1);
                int offset = id * sizeof(void *);
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

        builder->addReturn();
        auto const &insts = builder->getResult();

#ifdef ZFX_PRINT_IR
        printf("variables: %d slots\n", prog->locals.size());
        printf("channels: %d pointers\n", prog->chptrs.size());
        printf("consts:");
        for (auto const &val: prog->consts) printf(" %f", val);
        printf("\ninsts:");
        for (auto const &inst: insts) printf(" %02X", inst);
        printf("\n");
#endif

        prog->executable = std::make_unique<ExecutableInstance>(insts);
    }
};

std::unique_ptr<Program> Program::assemble(std::string const &lines) {
    Assembler a;
    a.parse(lines);
    return std::move(a.prog);
}

}
