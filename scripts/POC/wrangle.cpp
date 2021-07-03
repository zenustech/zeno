#include <vector>
#include <type_traits>
#include <magic_enum.hpp>
#include <iostream>
#include <array>

using std::cout;
using std::endl;

/* meta.h */

template <int First, int Last, typename Lambda>
inline constexpr bool static_for(Lambda const &f) {
    if constexpr (First < Last) {
        if (f(std::integral_constant<int, First>{})) {
            return true;
        } else {
            return static_for<First + 1, Last>(f);
        }
    }
    return false;
}

template <class...>
struct type_list {
};

template <>
struct type_list<> {
    static constexpr int length = 0;
};

template <class T, class ...Ts>
struct type_list<T, Ts...> {
    using head = T;
    using rest = type_list<Ts...>;
    static constexpr int length = rest::length + 1;
};

template <class L, unsigned int N>
struct type_list_nth {
    using type = typename type_list_nth<typename L::rest, N - 1>::type;
};

template <class L>
struct type_list_nth<L, 0> {
    using type = typename L::head;
};

template <class L, class T>
struct type_list_find {
    static constexpr int value = type_list_find<typename L::rest, T>::value + 1;
};

template <class T, class ...Ts>
struct type_list_find<type_list<T, Ts...>, T> {
    static constexpr int value = 0;
};

/* wrangle.h */

struct Context {
    float regtable[256];
    float *memtable[256];

    float memfetch(int index) {
        return *memtable[index];
    }

    void memstore(int index, float value) {
        *memtable[index] = value;
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
        case Opcode::add: z = x + y; break;
        case Opcode::sub: z = x - y; break;
        case Opcode::mul: z = x * y; break;
        case Opcode::div: z = x / y; break;
        }
        dst.set(ctx, z);
    }
};

/* main.cpp */

int main(void)
{
    Context ctx;
    Instruction inst;
    inst.opcode = Opcode::add;
    inst.dst.type = OperandType::reg;
    inst.dst.index = 0;
    inst.lhs.type = OperandType::reg;
    inst.lhs.index = 0;
    inst.rhs.type = OperandType::imm;
    inst.rhs.value = 3.14f;
    inst.execute(&ctx);
    cout << ctx.regtable[0] << endl;

    return 0;
}
