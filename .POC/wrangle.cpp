#include <vector>
#include <type_traits>
#include <magic_enum.hpp>
#include <iostream>
#include <cassert>
#include <sstream>
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

/* program.h */

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

/* wrangle.h */

void vectors_wrangle(Program const &prog,
    std::vector<std::vector<float> *> const &arrs) {
    if (arrs.size() == 0)
        return;
    size_t size = arrs[0]->size();
    for (int i = 1; i < arrs.size(); i++) {
        size = std::min(arrs[i]->size(), size);
    }
    Context ctx;
    for (int i = 0; i < arrs.size(); i++) {
        ctx.memtable[i] = arrs[i]->data();
    }
    for (int i = 0; i < size; i++) {
        ctx.memidx = i;
        prog.execute(&ctx);
        /*for (int j = 0; j < arrs.size(); j++) {
            ctx.memtable[j]++;
        }*/
    }
}

/* parse.cpp */

static std::vector<std::string> split_str(std::string const &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream iss(s);
  while (std::getline(iss, token, delimiter))
    tokens.push_back(token);
  return tokens;
}

Operand parse_operand(std::string const &ident) {
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

Instruction parse_instruction(std::string const &line) {
    auto tokens = split_str(line, ' ');
    assert(tokens.size() == 4);
    Instruction inst;
    inst.opcode = magic_enum::enum_cast<Opcode>(tokens[0]).value();
    inst.dst = parse_operand(tokens[1]);
    inst.lhs = parse_operand(tokens[2]);
    inst.rhs = parse_operand(tokens[3]);
    return inst;
}

Program parse_program(std::string const &lines) {
    Program prog;
    for (auto const &line: split_str(lines, '\n')) {
        if (line.size() != 0)
            prog.insts.push_back(parse_instruction(line));
    }
    return prog;
}

/* main.cpp */

int main(void)
{
    Context ctx;

    Program prog = parse_program(
        "add @0 @0 #3.14\n"
    );

    std::vector<float> arr(16);
    for (int i = 0; i < 8; i++) {
        arr[i] = 2.718f;
    }
    vectors_wrangle(prog, {&arr});

    for (int i = 0; i < 16; i++) {
        cout << arr[i] << endl;
    }

    return 0;
}
