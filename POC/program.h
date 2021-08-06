#pragma once

#include <vector>
#include <magic_enum.hpp>

struct Context {
    float regtable[256];
    float *memtable[256];
    size_t memindex;

    float memfetch(int index) const {
        return memtable[index][memindex];
    }

    void memstore(int index, float value) {
        memtable[index][memindex] = value;
    }
};

enum class Opcode : int {
    mov, add, sub, mul, div,
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

    float get(Context *ctx) const {
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

    void set(Context *ctx, float x) const {
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

    void execute(Context *ctx) const {
        float x = lhs.get(ctx);
        float y = rhs.get(ctx);
        float z = 0;
        switch (opcode) {
        case Opcode::mov: z = x; break;
        case Opcode::add: z = x + y; break;
        case Opcode::sub: z = x - y; break;
        case Opcode::mul: z = x * y; break;
        case Opcode::div: z = x / y; break;
        }
        dst.set(ctx, z);
    }
};

struct Program {
    std::vector<Instruction> insts;

    void execute(Context *ctx) const {
        for (auto const &inst: insts) {
            inst.execute(ctx);
        }
    }
};
