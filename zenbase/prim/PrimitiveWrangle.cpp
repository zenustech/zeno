#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/NumericObject.h>
#include <zen/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <stack>

namespace zenbase {


struct Opcode {
    enum {
        OP_LOAD,
        OP_STORE,
        OP_IMMED,

        OP_ADD,
        OP_SUB,
        OP_MUL,
        OP_DIV,
        OP_MOD,
        OP_POW,
        OP_ATAN2,
        OP_MIN,
        OP_MAX,
        OP_FMOD,
        OP_DOT,
        OP_CROSS,

        OP_NEG,
        OP_SQRT,
        OP_SIN,
        OP_COS,
        OP_TAN,
        OP_ASIN,
        OP_ACOS,
        OP_ATAN,
        OP_EXP,
        OP_LOG,
        OP_FLOOR,
        OP_CEIL,
        OP_LENGTH,
    };

    std::vector<int> ops;
    std::vector<int> pids;
    std::vector<float> imms;
    std::vector<std::string> names;

    Opcode() {
        ops.push_back(OP_LOAD);
        pids.push_back(0);
        names.push_back("pos");

        ops.push_back(OP_LENGTH);

        ops.push_back(OP_IMMED);
        imms.push_back(1.0);

        ops.push_back(OP_MUL);

        ops.push_back(OP_STORE);
        pids.push_back(0);
        names.push_back("clr");
    }

    void apply(size_t index, std::vector<PrimitiveObject *> const &primList) const {
        auto opit = ops.begin();
        auto pidit = pids.begin();
        auto immit = imms.begin();
        auto nameit = names.begin();
        std::stack<zen::vec3f> stack;
        for (; opit != ops.end(); opit++) {
            switch (*opit) {
                case OP_LOAD: {
                    auto const &arr = primList[*pidit++]->attr<zen::vec3f>(*nameit++);
                    stack.push(arr[index]);
                } break;
                case OP_STORE: {
                    auto &arr = primList[*pidit++]->attr<zen::vec3f>(*nameit++);
                    arr[index] = stack.top(); stack.pop();
                } break;
                case OP_IMMED: {
                    auto val = zen::vec3f(*immit++);
                    stack.push(val);
                } break;

                #define _PER_BINARY_OP(op, expr) \
                case op: { \
                    auto &lhs = stack.top(); stack.pop(); \
                    auto &rhs = stack.top(); stack.pop(); \
                    auto ret = (expr); \
                    stack.push(ret); \
                } break;
                _PER_BINARY_OP(OP_ADD, lhs + rhs)
                _PER_BINARY_OP(OP_SUB, lhs - rhs)
                _PER_BINARY_OP(OP_MUL, lhs * rhs)
                _PER_BINARY_OP(OP_DIV, lhs / rhs)
                _PER_BINARY_OP(OP_POW, zen::pow(lhs, rhs))
                _PER_BINARY_OP(OP_ATAN2, zen::atan2(lhs, rhs))
                _PER_BINARY_OP(OP_MIN, zen::min(lhs, rhs))
                _PER_BINARY_OP(OP_MAX, zen::max(lhs, rhs))
                _PER_BINARY_OP(OP_FMOD, zen::fmod(lhs, rhs))
                _PER_BINARY_OP(OP_DOT, zen::vec3f(zen::dot(lhs, rhs)))
                _PER_BINARY_OP(OP_CROSS, zen::cross(lhs, rhs))
                #undef _PER_BINARY_OP

                #define _PER_UNARY_OP(op, expr) \
                case op: { \
                    auto &lhs = stack.top(); stack.pop(); \
                    auto ret = (expr); \
                    stack.push(ret); \
                } break;
                _PER_UNARY_OP(OP_NEG, -lhs)
                _PER_UNARY_OP(OP_SQRT, zen::sqrt(lhs))
                _PER_UNARY_OP(OP_SIN, zen::sin(lhs))
                _PER_UNARY_OP(OP_COS, zen::cos(lhs))
                _PER_UNARY_OP(OP_TAN, zen::tan(lhs))
                _PER_UNARY_OP(OP_ASIN, zen::asin(lhs))
                _PER_UNARY_OP(OP_ACOS, zen::acos(lhs))
                _PER_UNARY_OP(OP_ATAN, zen::atan(lhs))
                _PER_UNARY_OP(OP_EXP, zen::exp(lhs))
                _PER_UNARY_OP(OP_LOG, zen::log(lhs))
                _PER_UNARY_OP(OP_FLOOR, zen::floor(lhs))
                _PER_UNARY_OP(OP_CEIL, zen::ceil(lhs))
                _PER_UNARY_OP(OP_LENGTH, zen::vec3f(zen::length(lhs)))
                #undef _PER_UNARY_OP
            }
        }
    }
};


struct PrimitiveWrangle : zen::INode {
  virtual void apply() override {
    std::vector<PrimitiveObject *> primList = {
        has_input("primA") ? get_input("primA")->as<PrimitiveObject>() : nullptr,
        has_input("primB") ? get_input("primB")->as<PrimitiveObject>() : nullptr,
        has_input("primC") ? get_input("primC")->as<PrimitiveObject>() : nullptr,
        has_input("primD") ? get_input("primD")->as<PrimitiveObject>() : nullptr,
    };
    assert(primList[0]);

    Opcode opcode;

    #pragma omp parallel for
    for (size_t i = 0; i < primList[0]->size(); i++) {
        opcode.apply(i, primList);
    }

    set_output_ref("primA", get_input_ref("primA"));
  }
};

static int defPrimitiveWrangle = zen::defNodeClass<PrimitiveWrangle>("PrimitiveWrangle",
    { /* inputs: */ {
    "primA",
    "primB",
    "primC",
    "primD",
    }, /* outputs: */ {
    "primA",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});

}
