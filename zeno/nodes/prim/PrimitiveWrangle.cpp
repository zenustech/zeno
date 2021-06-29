#include <zeno/zen.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <cctype>
#include <stack>
#include <iostream>

#define ARR_SKIP 1


namespace zen {

template <class T, size_t N>
struct array : std::array<T, N> {
    array() = default;

    static array fill(T const &x) {
        array a;
        for (size_t i = 0; i < N; i++) {
            a[i] = x;
        }
        return a;
    }

    static array load(T const *x) {
        array a;
        for (size_t i = 0; i < N; i++) {
            a[i] = x[i];
        }
        return a;
    }

    static void store(T *x, array const &a) {
        for (size_t i = 0; i < N; i++) {
            x[i] = a[i];
        }
    }

    template <class F>
    static array apply(F const &f, array const &a) {
        array r;
        for (size_t i = 0; i < N; i++) {
            r[i] = f(a[i]);
        }
        return r;
    }

    template <class F>
    static array apply(F const &f, array const &a, array const &b) {
        array r;
        for (size_t i = 0; i < N; i++) {
            r[i] = f(a[i], b[i]);
        }
        return r;
    }

    template <class F>
    static array apply(F const &f, array const &a, array const &b, array const &c) {
        array r;
        for (size_t i = 0; i < N; i++) {
            r[i] = f(a[i], b[i], c[i]);
        }
        return r;
    }
};

};


namespace zen {


struct Opcode : zen::IObject {
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
        OP_NORMALIZE,
        OP_DOTX,
        OP_DOTY,
        OP_DOTZ,

        OP_VEC,
        OP_MIX,
        OP_MLA,
    };

    std::vector<int> ops;
    std::vector<int> pids;
    std::vector<zen::vec3f> imms;
    std::vector<std::string> names;

    static int opFromName(std::string name) {
        std::transform(name.begin(), name.end(), name.begin(), ::toupper);
        if (0) {
        #define _EVAL(x) x
        #define _PER_NAME(x) \
        } else if (name == #x) { \
            return _EVAL(OP_##x);
        _PER_NAME(ADD)
        _PER_NAME(SUB)
        _PER_NAME(MUL)
        _PER_NAME(DIV)
        _PER_NAME(MOD)
        _PER_NAME(POW)
        _PER_NAME(ATAN2)
        _PER_NAME(MIN)
        _PER_NAME(MAX)
        _PER_NAME(FMOD)
        _PER_NAME(DOT)
        _PER_NAME(CROSS)

        _PER_NAME(NEG)
        _PER_NAME(SQRT)
        _PER_NAME(SIN)
        _PER_NAME(COS)
        _PER_NAME(TAN)
        _PER_NAME(ASIN)
        _PER_NAME(ACOS)
        _PER_NAME(ATAN)
        _PER_NAME(EXP)
        _PER_NAME(LOG)
        _PER_NAME(FLOOR)
        _PER_NAME(CEIL)
        _PER_NAME(LENGTH)
        _PER_NAME(NORMALIZE)
        _PER_NAME(DOTX)
        _PER_NAME(DOTY)
        _PER_NAME(DOTZ)

        _PER_NAME(VEC)
        _PER_NAME(MIX)
        _PER_NAME(MLA)
        #undef _PER_NAME
        #undef _EVAL
        }
        assert(0 && "bad op name");
        return -1;
    }

    void concat(Opcode const &other) {
        for (auto const &i: other.ops) {
            ops.push_back(i);
        }
        for (auto const &i: other.pids) {
            pids.push_back(i);
        }
        for (auto const &i: other.imms) {
            imms.push_back(i);
        }
        for (auto const &i: other.names) {
            names.push_back(i);
        }
    }

    void apply(size_t index, std::vector<PrimitiveObject *> const &primList) const {
        auto opit = ops.begin();
        auto pidit = pids.begin();
        auto immit = imms.begin();
        auto nameit = names.begin();
        using ValType = zen::array<zen::vec3f, ARR_SKIP>;
        std::stack<ValType> stack;
        for (; opit != ops.end(); opit++) {
            switch (*opit) {
                case OP_LOAD: {
                    auto const &arr = primList[*pidit++]->attr<zen::vec3f>(*nameit++);
                    stack.push(ValType::load(arr.data() + index));
                } break;
                case OP_STORE: {
                    auto &arr = primList[*pidit++]->attr<zen::vec3f>(*nameit++);
                    auto val = stack.top(); stack.pop();
                    ValType::store(arr.data() + index, val);
                } break;
                case OP_IMMED: {
                    stack.push(ValType::fill(*immit++));
                } break;

                #define _PER_BINARY_OP(op, expr) \
                case op: { \
                    auto &rhs = stack.top(); stack.pop(); \
                    auto &lhs = stack.top(); stack.pop(); \
                    auto ret = ValType::apply([]( \
                            auto const &lhs, auto const &rhs) { \
                        return (expr); \
                    }, lhs, rhs); \
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
                    auto ret = ValType::apply([]( \
                            auto const &lhs) { \
                        return (expr); \
                    }, lhs); \
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
                _PER_UNARY_OP(OP_NORMALIZE, zen::normalize(lhs))
                _PER_UNARY_OP(OP_DOTX, zen::vec3f(lhs[0]))
                _PER_UNARY_OP(OP_DOTY, zen::vec3f(lhs[1]))
                _PER_UNARY_OP(OP_DOTZ, zen::vec3f(lhs[2]))
                #undef _PER_UNARY_OP

                #define _PER_TERNARY_OP(op, expr) \
                case op: { \
                    auto &rhs = stack.top(); stack.pop(); \
                    auto &mhs = stack.top(); stack.pop(); \
                    auto &lhs = stack.top(); stack.pop(); \
                    auto ret = ValType::apply([]( \
                            auto const &lhs, auto const &mhs, auto const &rhs) { \
                        return (expr); \
                    }, lhs, mhs, rhs); \
                    stack.push(ret); \
                } break;
                _PER_TERNARY_OP(OP_VEC, zen::vec3f(lhs[0], mhs[0], rhs[0]))
                _PER_TERNARY_OP(OP_MIX, zen::mix(lhs, mhs, rhs))
                _PER_TERNARY_OP(OP_MLA, lhs * mhs + rhs)
                #undef _PER_TERNARY_OP
            }
        }
    }
};


struct WrangleImmed : zen::INode {
  virtual void apply() override {
    auto value = get_input("value")->as<NumericObject>();
    auto res = std::make_shared<Opcode>();
    zen::vec3f val;
    if (value->is<float>())
        val = zen::vec3f(value->get<float>());
    else
        val = value->get<zen::vec3f>();
    res->ops.push_back(Opcode::OP_IMMED);
    res->imms.push_back(val);
    set_output("res", std::move(res));
  }
};

static int defWrangleImmed = zen::defNodeClass<WrangleImmed>("WrangleImmed",
    { /* inputs: */ {
    "value",
    }, /* outputs: */ {
    "res",
    }, /* params: */ {
    }, /* category: */ {
    "wrangler",
    }});


struct WrangleLoad : zen::INode {
  virtual void apply() override {
    auto primId = std::get<int>(get_param("primId"));
    auto attrName = std::get<std::string>(get_param("attrName"));
    auto res = std::make_shared<Opcode>();
    res->ops.push_back(Opcode::OP_LOAD);
    res->pids.push_back(primId);
    res->names.push_back(attrName);
    set_output("res", std::move(res));
  }
};

static int defWrangleLoad = zen::defNodeClass<WrangleLoad>("WrangleLoad",
    { /* inputs: */ {
    }, /* outputs: */ {
    "res",
    }, /* params: */ {
    {"int", "primId", "0"},
    {"string", "attrName", "pos"},
    }, /* category: */ {
    "wrangler",
    }});


struct WrangleStore : zen::INode {
  virtual void apply() override {
    auto primId = std::get<int>(get_param("primId"));
    auto attrName = std::get<std::string>(get_param("attrName"));
    auto res = std::make_shared<Opcode>();
    auto val = get_input("val")->as<Opcode>();
    res->concat(*val);
    res->ops.push_back(Opcode::OP_STORE);
    res->pids.push_back(primId);
    res->names.push_back(attrName);
    set_output("res", res);
  }
};

static int defWrangleStore = zen::defNodeClass<WrangleStore>("WrangleStore",
    { /* inputs: */ {
    "val",
    }, /* outputs: */ {
    "res",
    }, /* params: */ {
    {"int", "primId", "0"},
    {"string", "attrName", "pos"},
    }, /* category: */ {
    "wrangler",
    }});


struct WrangleUnaryOp : zen::INode {
  virtual void apply() override {
    auto opName = std::get<std::string>(get_param("opName"));
    auto res = std::make_shared<Opcode>();
    auto lhs = get_input("lhs")->as<Opcode>();
    res->concat(*lhs);
    res->ops.push_back(Opcode::opFromName(opName));
    set_output("res", res);
  }
};

static int defWrangleUnaryOp = zen::defNodeClass<WrangleUnaryOp>("WrangleUnaryOp",
    { /* inputs: */ {
    "lhs",
    }, /* outputs: */ {
    "res",
    }, /* params: */ {
    {"string", "opName", "neg"},
    }, /* category: */ {
    "wrangler",
    }});


struct WrangleBinaryOp : zen::INode {
  virtual void apply() override {
    auto opName = std::get<std::string>(get_param("opName"));
    auto res = std::make_shared<Opcode>();
    auto lhs = get_input("lhs")->as<Opcode>();
    auto rhs = get_input("rhs")->as<Opcode>();
    res->concat(*lhs);
    res->concat(*rhs);
    res->ops.push_back(Opcode::opFromName(opName));
    set_output("res", res);
  }
};

static int defWrangleBinaryOp = zen::defNodeClass<WrangleBinaryOp>("WrangleBinaryOp",
    { /* inputs: */ {
    "lhs",
    "rhs",
    }, /* outputs: */ {
    "res",
    }, /* params: */ {
    {"string", "opName", "add"},
    }, /* category: */ {
    "wrangler",
    }});


struct WrangleTernaryOp : zen::INode {
  virtual void apply() override {
    auto opName = std::get<std::string>(get_param("opName"));
    auto res = std::make_shared<Opcode>();
    auto lhs = get_input("lhs")->as<Opcode>();
    auto mhs = get_input("mhs")->as<Opcode>();
    auto rhs = get_input("rhs")->as<Opcode>();
    res->concat(*lhs);
    res->concat(*mhs);
    res->concat(*rhs);
    res->ops.push_back(Opcode::opFromName(opName));
    set_output("res", res);
  }
};

static int defWrangleTernaryOp = zen::defNodeClass<WrangleTernaryOp>("WrangleTernaryOp",
    { /* inputs: */ {
    "lhs",
    "mhs",
    "rhs",
    }, /* outputs: */ {
    "res",
    }, /* params: */ {
    {"string", "opName", "vec"},
    }, /* category: */ {
    "wrangler",
    }});


struct PrimitiveWrangle : zen::INode {
  virtual void apply() override {
    std::vector<PrimitiveObject *> primList = {
        has_input("prim0") ? get_input("prim0")->as<PrimitiveObject>() : nullptr,
        has_input("prim1") ? get_input("prim1")->as<PrimitiveObject>() : nullptr,
        has_input("prim2") ? get_input("prim2")->as<PrimitiveObject>() : nullptr,
        has_input("prim3") ? get_input("prim3")->as<PrimitiveObject>() : nullptr,
    };
    assert(primList[0]);

    auto opcode = get_input("wrangle")->as<Opcode>();

    #pragma omp parallel for
    for (int i = 0; i < primList[0]->size(); i += ARR_SKIP) {
        opcode->apply(i, primList);
    }

    set_output_ref("prim0", get_input_ref("prim0"));
  }
};

static int defPrimitiveWrangle = zen::defNodeClass<PrimitiveWrangle>("PrimitiveWrangle",
    { /* inputs: */ {
    "wrangle",
    "prim0",
    "prim1",
    "prim2",
    "prim3",
    }, /* outputs: */ {
    "prim0",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});

}
