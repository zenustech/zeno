#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/NumericObject.h>
#include <zen/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zenbase {

template <class T, class S>
inline constexpr bool is_decay_same_v = std::is_same_v<std::decay_t<T>, std::decay_t<S>>;


template <class FuncT>
struct UnaryOperator {
    FuncT func;
    UnaryOperator(FuncT const &func) : func(func) {}

    template <class TOut, class TA>
    void operator()(std::vector<TOut> &arrOut, std::vector<TA> const &arrA) {
        size_t n = std::min(arrOut.size(), arrA.size());
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            auto val = func(arrA[i]);
            arrOut[i] = (decltype(arrOut[0]))val;
        }
    }
};

struct PrimitiveUnaryOp : zen::INode {
  virtual void apply() override {
    auto primA = get_input("primA")->as<PrimitiveObject>();
    auto primOut = get_input("primOut")->as<PrimitiveObject>();
    auto attrA = std::get<std::string>(get_param("attrA"));
    auto attrOut = std::get<std::string>(get_param("attrOut"));
    auto op = std::get<std::string>(get_param("op"));
    auto const &arrA = primA->attr(attrA);
    auto &arrOut = primOut->attr(attrOut);
    std::visit([op](auto &arrOut, auto const &arrA) {
        if constexpr (zen::is_vec_castable_v<decltype(arrOut[0]), decltype(arrA[0])>) {
            if (0) {
#define _PER_OP(opname, expr) \
            } else if (op == opname) { \
                UnaryOperator([](auto const &a) { return expr; })(arrOut, arrA);
            _PER_OP("copy", a)
            _PER_OP("neg", -a)
            _PER_OP("sqrt", zen::sqrt(a))
            _PER_OP("sin", zen::sin(a))
            _PER_OP("cos", zen::cos(a))
            _PER_OP("tan", zen::tan(a))
            _PER_OP("asin", zen::asin(a))
            _PER_OP("acos", zen::acos(a))
            _PER_OP("atan", zen::atan(a))
            _PER_OP("exp", zen::exp(a))
            _PER_OP("log", zen::log(a))
#undef _PER_OP
            } else {
                printf("%s\n", op.c_str());
                assert(0 && "Bad operator type");
            }
        } else {
            assert(0 && "Failed to promote variant type");
        }
    }, arrOut, arrA);

    set_output_ref("primOut", get_input_ref("primOut"));
  }
};

static int defPrimitiveUnaryOp = zen::defNodeClass<PrimitiveUnaryOp>("PrimitiveUnaryOp",
    { /* inputs: */ {
    "primA",
    "primOut",
    }, /* outputs: */ {
    "primOut",
    }, /* params: */ {
    {"string", "attrA", "pos"},
    {"string", "attrOut", "pos"},
    {"string", "op", "copy"},
    }, /* category: */ {
    "primitive",
    }});


template <class FuncT>
struct BinaryOperator {
    FuncT func;
    BinaryOperator(FuncT const &func) : func(func) {}

    template <class TOut, class TA, class TB>
    void operator()(std::vector<TOut> &arrOut,
        std::vector<TA> const &arrA, std::vector<TB> const &arrB) {
        size_t n = std::min(arrOut.size(), std::min(arrA.size(), arrB.size()));
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            auto val = func(arrA[i], arrB[i]);
            arrOut[i] = (decltype(arrOut[0]))val;
        }
    }
};

struct PrimitiveBinaryOp : zen::INode {
  virtual void apply() override {
    auto primA = get_input("primA")->as<PrimitiveObject>();
    auto primB = get_input("primB")->as<PrimitiveObject>();
    auto primOut = get_input("primOut")->as<PrimitiveObject>();
    auto attrA = std::get<std::string>(get_param("attrA"));
    auto attrB = std::get<std::string>(get_param("attrB"));
    auto attrOut = std::get<std::string>(get_param("attrOut"));
    auto op = std::get<std::string>(get_param("op"));
    auto const &arrA = primA->attr(attrA);
    auto const &arrB = primB->attr(attrB);
    auto &arrOut = primOut->attr(attrOut);
    std::visit([op](auto &arrOut, auto const &arrA, auto const &arrB) {
        if constexpr (is_decay_same_v<decltype(arrOut[0]),
            zen::is_vec_promotable_t<decltype(arrA[0]), decltype(arrB[0])>>) {
            if (0) {
#define _PER_OP(opname, expr) \
            } else if (op == opname) { \
                BinaryOperator([](auto const &a_, auto const &b_) { \
                    using PromotedType = decltype(a_ + b_); \
                    auto a = PromotedType(a_); \
                    auto b = PromotedType(b_); \
                    return expr; \
                })(arrOut, arrA, arrB);
            _PER_OP("copyA", a)
            _PER_OP("copyB", b)
            _PER_OP("add", a + b)
            _PER_OP("sub", a - b)
            _PER_OP("rsub", b - a)
            _PER_OP("mul", a * b)
            _PER_OP("div", a / b)
            _PER_OP("rdiv", b / a)
            _PER_OP("pow", zen::pow(a, b))
            _PER_OP("rpow", zen::pow(b, a))
            _PER_OP("atan2", zen::atan2(a, b))
            _PER_OP("ratan2", zen::atan2(b, a))
#undef _PER_OP
            } else {
                printf("%s\n", op.c_str());
                assert(0 && "Bad operator type");
            }
        } else {
            assert(0 && "Failed to promote variant type");
        }
    }, arrOut, arrA, arrB);

    set_output_ref("primOut", get_input_ref("primOut"));
  }
};

static int defPrimitiveBinaryOp = zen::defNodeClass<PrimitiveBinaryOp>("PrimitiveBinaryOp",
    { /* inputs: */ {
    "primA",
    "primB",
    "primOut",
    }, /* outputs: */ {
    "primOut",
    }, /* params: */ {
    {"string", "attrA", "pos"},
    {"string", "attrB", "pos"},
    {"string", "attrOut", "pos"},
    {"string", "op", "copyA"},
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveMix : zen::INode {
    virtual void apply() override{
        auto primA = get_input("primA")->as<PrimitiveObject>();
        auto primB = get_input("primB")->as<PrimitiveObject>();
        auto primOut = get_input("primOut")->as<PrimitiveObject>();
        auto attrA = std::get<std::string>(get_param("attrA"));
        auto attrB = std::get<std::string>(get_param("attrB"));
        auto attrOut = std::get<std::string>(get_param("attrOut"));
        auto const &arrA = primA->attr(attrA);
        auto const &arrB = primB->attr(attrB);
        auto &arrOut = primOut->attr(attrOut);
        auto coef = get_input("coef")->as<zenbase::NumericObject>()->get<float>();
        
        std::visit([coef](auto &arrA, auto &arrB, auto &arrOut) {
          if constexpr (std::is_same_v<decltype(arrA), decltype(arrB)> && std::is_same_v<decltype(arrA), decltype(arrOut)>) {
#pragma omp parallel for
            for (int i = 0; i < arrOut.size(); i++) {
                arrOut[i] = (1.0-coef)*arrA[i] + coef*arrB[i];
            }
          }
        }, arrA, arrB, arrOut);
        set_output_ref("primOut", get_input_ref("primOut"));
    }
};
static int defPrimitiveMix = zen::defNodeClass<PrimitiveMix>("PrimitiveMix",
    { /* inputs: */ {
    "primA",
    "primB",
    "primOut",
    "coef",
    }, /* outputs: */ {
    "primOut",
    }, /* params: */ {
    {"string", "attrA", "pos"},
    {"string", "attrB", "pos"},
    {"string", "attrOut", "pos"},
    }, /* category: */ {
    "primitive",
    }});


template <class FuncT>
struct HalfBinaryOperator {
    FuncT func;
    HalfBinaryOperator(FuncT const &func) : func(func) {}

    template <class TOut, class TA, class TB>
    void operator()(std::vector<TOut> &arrOut,
        std::vector<TA> const &arrA, TB const &valB) {
        size_t n = std::min(arrOut.size(), arrA.size());
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            auto val = func(arrA[i], valB);
            arrOut[i] = (decltype(arrOut[0]))val;
        }
    }
};

struct PrimitiveHalfBinaryOp : zen::INode {
  virtual void apply() override {
    auto primA = get_input("primA")->as<PrimitiveObject>();
    auto primOut = get_input("primOut")->as<PrimitiveObject>();
    auto attrA = std::get<std::string>(get_param("attrA"));
    auto attrOut = std::get<std::string>(get_param("attrOut"));
    auto op = std::get<std::string>(get_param("op"));
    auto const &arrA = primA->attr(attrA);
    auto &arrOut = primOut->attr(attrOut);
    auto const &valB = get_input("valueB")->as<NumericObject>()->value;
    std::visit([op](auto &arrOut, auto const &arrA, auto const &valB) {
        if constexpr (is_decay_same_v<decltype(arrOut[0]),
            zen::is_vec_promotable_t<decltype(arrA[0]), decltype(valB)>>) {
            if (0) {
#define _PER_OP(opname, expr) \
            } else if (op == opname) { \
                HalfBinaryOperator([](auto const &a_, auto const &b_) { \
                    using PromotedType = decltype(a_ + b_); \
                    auto a = PromotedType(a_); \
                    auto b = PromotedType(b_); \
                    return expr; \
                })(arrOut, arrA, valB);
            _PER_OP("copyA", a)
            _PER_OP("copyB", b)
            _PER_OP("add", a + b)
            _PER_OP("sub", a - b)
            _PER_OP("rsub", b - a)
            _PER_OP("mul", a * b)
            _PER_OP("div", a / b)
            _PER_OP("rdiv", b / a)
            _PER_OP("pow", zen::pow(a, b))
            _PER_OP("rpow", zen::pow(b, a))
            _PER_OP("atan2", zen::atan2(a, b))
            _PER_OP("ratan2", zen::atan2(b, a))
#undef _PER_OP
            } else {
                printf("%s\n", op.c_str());
                assert(0 && "Bad operator type");
            }
        } else {
            assert(0 && "Failed to promote variant type");
        }
    }, arrOut, arrA, valB);

    set_output_ref("primOut", get_input_ref("primOut"));
  }
};

static int defPrimitiveHalfBinaryOp = zen::defNodeClass<PrimitiveHalfBinaryOp>("PrimitiveHalfBinaryOp",
    { /* inputs: */ {
    "primA",
    "valueB",
    "primOut",
    }, /* outputs: */ {
    "primOut",
    }, /* params: */ {
    {"string", "attrA", "pos"},
    {"string", "attrOut", "pos"},
    {"string", "op", "copyA"},
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveFillAttr : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto const &value = get_input("value")->as<NumericObject>()->value;
    auto attrName = std::get<std::string>(get_param("attrName"));
    auto &arr = prim->attr(attrName);
    std::visit([](auto &arr, auto const &value) {
        if constexpr (zen::is_vec_castable_v<decltype(arr[0]), decltype(value)>) {
            #pragma omp parallel for
            for (int i = 0; i < arr.size(); i++) {
                arr[i] = decltype(arr[i])(value);
            }
        } else {
            assert(0 && "Failed to promote variant type");
        }
    }, arr, value);

    set_output_ref("prim", get_input_ref("prim"));
  }
};

static int defPrimitiveFillAttr = zen::defNodeClass<PrimitiveFillAttr>("PrimitiveFillAttr",
    { /* inputs: */ {
    "prim",
    "value",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"string", "attrName", "pos"},
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveRandomizeAttr : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto min = std::get<float>(get_param("min"));
    auto minY = std::get<float>(get_param("minY"));
    auto minZ = std::get<float>(get_param("minZ"));
    auto max = std::get<float>(get_param("max"));
    auto maxY = std::get<float>(get_param("maxY"));
    auto maxZ = std::get<float>(get_param("maxZ"));
    auto attrName = std::get<std::string>(get_param("attrName"));
    auto &arr = prim->attr(attrName);
    std::visit([min, minY, minZ, max, maxY, maxZ](auto &arr) {
        for (int i = 0; i < arr.size(); i++) {
            if constexpr (is_decay_same_v<decltype(arr[i]), zen::vec3f>) {
                zen::vec3f f(drand48(), drand48(), drand48());
                zen::vec3f a(min, minY, minZ);
                zen::vec3f b(max, maxY, maxZ);
                arr[i] = zen::mix(a, b, f);
            } else {
                arr[i] = zen::mix(min, max, (float)drand48());
            }
        }
    }, arr);

    set_output_ref("prim", get_input_ref("prim"));
  }
};

static int defPrimitiveRandomizeAttr = zen::defNodeClass<PrimitiveRandomizeAttr>("PrimitiveRandomizeAttr",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"string", "attrName", "pos"},
    {"float", "min", "-1"},
    {"float", "minY", "-1"},
    {"float", "minZ", "-1"},
    {"float", "max", "1"},
    {"float", "maxY", "1"},
    {"float", "maxZ", "1"},
    }, /* category: */ {
    "primitive",
    }});


void print_cout(float x) {
    printf("%f\n", x);
}

void print_cout(zen::vec3f const &a) {
    printf("%f %f %f\n", a[0], a[1], a[2]);
}


struct PrimitivePrintAttr : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto attrName = std::get<std::string>(get_param("attrName"));
    auto const &arr = prim->attr(attrName);
    std::visit([attrName](auto const &arr) {
        printf("attribute `%s`, length %d:\n", attrName.c_str(), arr.size());
        for (int i = 0; i < arr.size(); i++) {
            print_cout(arr[i]);
        }
        if (arr.size() == 0) {
            printf("(no data)\n");
        }
        printf("\n");
    }, arr);

    set_output_ref("prim", get_input_ref("prim"));
  }
};

static int defPrimitivePrintAttr = zen::defNodeClass<PrimitivePrintAttr>("PrimitivePrintAttr",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"string", "attrName", "pos"},
    }, /* category: */ {
    "primitive",
    }});

}
