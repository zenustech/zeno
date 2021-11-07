#include "SIMDBuilder.h"
#include "Executable.h"
#include "FuncTable.h"
#include <zfx/utils.h>
#include <zfx/x64.h>
#include <algorithm>
#include <sstream>
#include <map>

namespace zfx::x64 {

#define ERROR_IF(x) do { \
    if (x) { \
        error("`%s`", #x); \
    } \
} while (0)

struct ImplAssembler {
    int simdkind = simdtype::xmmps;

    std::unique_ptr<SIMDBuilder> builder = std::make_unique<SIMDBuilder>();
    std::unique_ptr<Executable> exec = std::make_unique<Executable>();
    static inline std::unique_ptr<FuncTable> functable;

    int nconsts = 0;
    int nlocals = 0;
    //int nglobals = 0;

    static float parse_float(std::string const &expr) {
        float value = 0.0f;
        if (std::istringstream(expr) >> value)
            return value;
        union {
            float f;
            uint32_t i;
        } u;
        u.i = 0xffffffff;
        return u.f;
    }

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
                exec->consts[id] = parse_float(expr);

            } else if (cmd == "ldp") {
                // rsi points to an array of constants
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nconsts = std::max(nconsts, id + 1);
                int offset = id * SIMDBuilder::scalarSizeOfType(simdkind);
                builder->addAvxBroadcastLoadOp(simdkind,
                    dst, { opreg::a2, memflag::reg_imm8, offset});

            } else if (cmd == "ldl") {
                // rdi points to an array of variables
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nlocals = std::max(nlocals, id + 1);
                int offset = id * SIMDBuilder::sizeOfType(simdkind);
                builder->addAvxMemoryOp(simdkind, opcode::loadu,
                    dst, {opreg::a1, memflag::reg_imm8, offset});

            } else if (cmd == "stl") {
                ERROR_IF(linesep.size() < 2);
                auto dst = from_string<int>(linesep[1]);
                auto id = from_string<int>(linesep[2]);
                nlocals = std::max(nlocals, id + 1);
                int offset = id * SIMDBuilder::sizeOfType(simdkind);
                builder->addAvxMemoryOp(simdkind, opcode::storeu,
                    dst, {opreg::a1, memflag::reg_imm8, offset});

            /*} else if (cmd == "ldg") {
                // rdx points to an array of pointers
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

            } else if (cmd == "min") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::min,
                    dst, lhs, rhs);

            } else if (cmd == "max") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::max,
                    dst, lhs, rhs);

            } else if (cmd == "and") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::bit_and,
                    dst, lhs, rhs);

            } else if (cmd == "andnot") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::bit_andn,
                    dst, rhs, lhs);

            } else if (cmd == "cmpeq") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::cmp_eq,
                    dst, lhs, rhs);

            } else if (cmd == "cmpne") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::cmp_ne,
                    dst, lhs, rhs);

            } else if (cmd == "cmplt") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::cmp_lt,
                    dst, lhs, rhs);

            } else if (cmd == "cmple") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::cmp_le,
                    dst, lhs, rhs);

            } else if (cmd == "cmpgt") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::cmp_gt,
                    dst, lhs, rhs);

            } else if (cmd == "cmpge") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::cmp_ge,
                    dst, lhs, rhs);

            } else if (cmd == "or") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::bit_or,
                    dst, lhs, rhs);

            } else if (cmd == "xor") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto lhs = from_string<int>(linesep[2]);
                auto rhs = from_string<int>(linesep[3]);
                builder->addAvxBinaryOp(simdkind, opcode::bit_xor,
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
                builder->addAvxMoveOp(simdkind, dst, src);
            } else if (cmd == "blend") {
            // } else if (cmd == "round") {
            //     ERROR_IF(linesep.size() < 2);
            //     auto dst = from_string<int>(linesep[1]);
            //     auto src = from_string<int>(linesep[2]);
            //     builder->addAvxRoundOp(simdkind, dst, src, 0 + 8);

            // } else if (cmd == "floor") {
            //     ERROR_IF(linesep.size() < 2);
            //     auto dst = from_string<int>(linesep[1]);
            //     auto src = from_string<int>(linesep[2]);
            //     builder->addAvxRoundOp(simdkind, dst, src, 1 + 8);

            // } else if (cmd == "ceil") {
            //     ERROR_IF(linesep.size() < 2);
            //     auto dst = from_string<int>(linesep[1]);
            //     auto src = from_string<int>(linesep[2]);
            //     builder->addAvxRoundOp(simdkind, dst, src, 2 + 8);

            // } else if (cmd == "blend") {
                ERROR_IF(linesep.size() < 3);
                auto dst = from_string<int>(linesep[1]);
                auto cond = from_string<int>(linesep[2]);
                auto lhs = from_string<int>(linesep[3]);
                auto rhs = from_string<int>(linesep[4]);
                builder->addAvxBlendvOp(simdkind, dst, rhs, lhs, cond);

            } else if (auto it = std::find(
                FuncTable::funcnames.begin(), FuncTable::funcnames.end(), cmd);
                it != FuncTable::funcnames.end()) {
                // rdx points to an array of function pointers
                ERROR_IF(linesep.size() < 2);
                if (linesep.size() == 3) {
                    auto dst = from_string<int>(linesep[1]);
                    auto src = from_string<int>(linesep[2]);
                    builder->addPushReg(opreg::a3);
                    builder->addPushReg(opreg::a2);
                    builder->addPushReg(opreg::a1);
                    int size = SIMDBuilder::sizeOfType(simdkind);
                    builder->addAdjStackTop(-size);
                    builder->addAvxMemoryOp(simdkind, opcode::storeu,
                        src, opreg::rsp);
                    builder->addRegularMoveOp(opreg::a1, opreg::rsp);
                    int id = it - FuncTable::funcnames.begin();
                    int offset = id * sizeof(void *);
#if defined(_WIN32)
                    builder->addAdjStackTop(-64);
#endif
                    builder->addCallOp({opreg::a3, memflag::reg_imm8, offset});
#if defined(_WIN32)
                    builder->addAdjStackTop(64);
#endif
                    builder->addAvxMemoryOp(simdkind, opcode::loadu,
                        dst, opreg::rsp);
                    builder->addAdjStackTop(size);
                    builder->addPopReg(opreg::a1);
                    builder->addPopReg(opreg::a2);
                    builder->addPopReg(opreg::a3);
                } else {
                    auto dst = from_string<int>(linesep[1]);
                    auto lhs = from_string<int>(linesep[2]);
                    auto rhs = from_string<int>(linesep[3]);
                    builder->addPushReg(opreg::a3);
                    builder->addPushReg(opreg::a2);
                    builder->addPushReg(opreg::a1);
                    int size = SIMDBuilder::sizeOfType(simdkind);
                    builder->addAdjStackTop(-size);
                    builder->addAvxMemoryOp(simdkind, opcode::storeu,
                        rhs, opreg::rsp);
                    builder->addRegularMoveOp(opreg::a2, opreg::rsp);
                    builder->addAdjStackTop(-size);
                    builder->addAvxMemoryOp(simdkind, opcode::storeu,
                        lhs, opreg::rsp);
                    builder->addRegularMoveOp(opreg::a1, opreg::rsp);
                    int id = it - FuncTable::funcnames.begin();
                    int offset = id * sizeof(void *);
#if defined(_WIN32)
                    builder->addAdjStackTop(-64);
#endif
                    builder->addCallOp({opreg::a3, memflag::reg_imm8, offset});
#if defined(_WIN32)
                    builder->addAdjStackTop(64);
#endif
                    builder->addAvxMemoryOp(simdkind, opcode::loadu,
                        dst, opreg::rsp);
                    builder->addAdjStackTop(size * 2);
                    builder->addPopReg(opreg::a1);
                    builder->addPopReg(opreg::a2);
                    builder->addPopReg(opreg::a3);
                }

            } else {
                error("bad assembly command `%s`", cmd.c_str());
            }
        }

        builder->addReturn();
        auto const &insts = builder->getResult();

#ifdef ZFX_PRINT_IR
        log_printf("variables: %d slots\n", nlocals);
        log_printf("consts: %d values\n", nconsts);
        {
            std::string _ = "insts:";
            for (auto const &inst: insts) {
                char buf[8];
                sprintf(buf, " %02X", inst);
                _ += buf;
            }
            log_printf("%s\n", _.c_str());
        }
#endif

        if (!functable)
            functable = std::make_unique<FuncTable>();
        exec->functable = functable->funcptrs.data();
        exec->memsize = (insts.size() + 4095) / 4096 * 4096;
        exec->mem = (uint8_t *)exec_page_allocate(exec->memsize);
        for (int i = 0; i < insts.size(); i++) {
            exec->mem[i] = insts[i];
        }
        exec_page_mark_executable(exec->mem, exec->memsize);
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
