#pragma once

#include <cstdio>
#include <vector>
#include <string>
#include <cmath>
#include <map>

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

std::string zfx_to_assembly(std::string const &code);
Program assemble_program(std::string const &lines);
Program *compile_program(std::string const &code);
