#include "SIMDBuilder.h"
#include "Executable.h"
#include <zfx/utils.h>
#include <zfx/x64.h>
#include <sstream>
#include <map>

namespace zfx::x64 {

#define ERROR_IF(x) do { \
    if (x) { \
        error("`%s`", #x); \
    } \
} while (0)

struct ImplAssembler {
    int simdkind = optype::xmmps;
    std::unique_ptr<SIMDBuilder> builder = std::make_unique<SIMDBuilder>();
    std::unique_ptr<Executable> exec = std::make_unique<Executable>();

    int nconsts = 0;
    int nlocals = 0;
    //int nglobals = 0;

    void parse(std::string const &lines) {
        for (auto line: split_str(lines, '\n')) {
            if (!line.size()) continue;

            auto linesep = split_str(line, ' ');
            ERROR_IF(linesep.size() < 1);
            auto cmd = linesep[0];
            if (cmd == "const") {
                ERROR_IF(linesep.size() < 2);
                auto id = from_string<int>(linesep[1]);
                auto expr = linesep[2];
                if (!(std::istringstream(expr) >> exec->consts[id])) {
                    error("cannot parse literial constant `%s`",
                        expr.c_str());
                }

            } else if (cmd == "ldi") {  // rcx points to an array of constants
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nconsts = std::max(nconsts, id + 1);
                int offset = id * SIMDBuilder::scalarSizeOfType(simdkind);
                builder->addAvxBroadcastLoadOp(simdkind,
                    dst, {opreg::rcx, memflag::reg_imm8, offset});

            } else if (cmd == "ldm") {  // rdx points to an array of variables
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nlocals = std::max(nlocals, id + 1);
                int offset = id * SIMDBuilder::sizeOfType(simdkind);
                builder->addAvxMemoryOp(simdkind, opcode::loadu,
                    dst, {opreg::rdx, memflag::reg_imm8, offset});

            } else if (cmd == "stm") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nlocals = std::max(nlocals, id + 1);
                int offset = id * SIMDBuilder::sizeOfType(simdkind);
                builder->addAvxMemoryOp(simdkind, opcode::storeu,
                    dst, {opreg::rdx, memflag::reg_imm8, offset});

            /*} else if (cmd == "ldg") {  // rdx points to an array of pointers
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nglobals = std::max(nglobals, id + 1);
                int offset = id * sizeof(void *);
                builder->addRegularLoadOp(opreg::rax,
                    {opreg::rdx, memflag::reg_imm8, offset});
                builder->addAvxMemoryOp(simdkind, opcode::loadu,
                    dst, opreg::rax);

            } else if (cmd == "stg") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nglobals = std::max(nglobals, id + 1);
                int offset = id * sizeof(void *);
                builder->addRegularLoadOp(opreg::rax,
                    {opreg::rdx, memflag::reg_imm8, offset});
                builder->addAvxMemoryOp(simdkind, opcode::storeu,
                    dst, opreg::rax);*/

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

            } else if (cmd == "sqrt") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto src = from_string<int>(linesep[2]);
                builder->addAvxUnaryOp(simdkind, opcode::sqrt,
                    dst, src);

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
        printf("variables: %d slots\n", nlocals);
        //printf("channels: %d pointers\n", nglobals);
        printf("consts: %d values\n", nconsts);
        printf("insts:");
        for (auto const &inst: insts) printf(" %02X", inst);
        printf("\n");
#endif

        exec->memsize = (insts.size() + 4095) / 4096 * 4096;
        exec->mem = exec_page_alloc(exec->memsize);
        for (int i = 0; i < insts.size(); i++) {
            exec->mem[i] = insts[i];
        }
    }
};

std::unique_ptr<Executable> Executable::assemble
    ( std::string const &lines
    ) {
    ImplAssembler a;
    a.parse(lines);
    return std::move(a.exec);
}

Executable::~Executable() {
    if (mem) {
        exec_page_free(mem, memsize);
        mem = nullptr;
        memsize = 0;
    }
}

}
