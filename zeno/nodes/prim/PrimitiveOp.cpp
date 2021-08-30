#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>

#ifdef _MSC_VER
static inline double drand48() {
	return rand() / (double)RAND_MAX;
}
#endif

namespace zeno {

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

struct PrimitiveUnaryOp : zeno::INode {
  virtual void apply() override {
    auto primA = get_input<PrimitiveObject>("primA");
    auto primOut = get_input<PrimitiveObject>("primOut");
    auto attrA = std::get<std::string>(get_param("attrA"));
    auto attrOut = std::get<std::string>(get_param("attrOut"));
    auto op = std::get<std::string>(get_param("op"));
    auto const &arrA = primA->attr(attrA);
    auto &arrOut = primOut->attr(attrOut);
    std::visit([op](auto &arrOut, auto const &arrA) {
        if constexpr (zeno::is_vec_castable_v<decltype(arrOut[0]), decltype(arrA[0])>) {
            if (0) {
#define _PER_OP(opname, expr) \
            } else if (op == opname) { \
                UnaryOperator([](auto const &a) { return expr; })(arrOut, arrA);
            _PER_OP("copy", a)
            _PER_OP("neg", -a)
            _PER_OP("sqrt", zeno::sqrt(a))
            _PER_OP("sin", zeno::sin(a))
            _PER_OP("cos", zeno::cos(a))
            _PER_OP("tan", zeno::tan(a))
            _PER_OP("asin", zeno::asin(a))
            _PER_OP("acos", zeno::acos(a))
            _PER_OP("atan", zeno::atan(a))
            _PER_OP("exp", zeno::exp(a))
            _PER_OP("log", zeno::log(a))
#undef _PER_OP
            } else {
                throw zeno::Exception("Bad operator type: " + op);
            }
        } else {
            throw zeno::Exception("Failed to promote variant type");
        }
    }, arrOut, arrA);

    set_output("primOut", get_input("primOut"));
  }
};

ZENDEFNODE(PrimitiveUnaryOp,
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

struct PrimitiveBinaryOp : zeno::INode {
  virtual void apply() override {
    auto primA = get_input<PrimitiveObject>("primA");
    auto primB = get_input<PrimitiveObject>("primB");
    auto primOut = get_input<PrimitiveObject>("primOut");
    auto attrA = std::get<std::string>(get_param("attrA"));
    auto attrB = std::get<std::string>(get_param("attrB"));
    auto attrOut = std::get<std::string>(get_param("attrOut"));
    auto op = std::get<std::string>(get_param("op"));
    auto const &arrA = primA->attr(attrA);
    auto const &arrB = primB->attr(attrB);
    auto &arrOut = primOut->attr(attrOut);
    std::visit([op](auto &arrOut, auto const &arrA, auto const &arrB) {
        if constexpr (is_decay_same_v<decltype(arrOut[0]),
            zeno::is_vec_promotable_t<decltype(arrA[0]), decltype(arrB[0])>>) {
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
            _PER_OP("pow", zeno::pow(a, b))
            _PER_OP("rpow", zeno::pow(b, a))
            _PER_OP("atan2", zeno::atan2(a, b))
            _PER_OP("ratan2", zeno::atan2(b, a))
#undef _PER_OP
            } else {
                throw zeno::Exception("Bad operator type: " + op);
            }
        } else {
            throw zeno::Exception("Failed to promote variant type");
        }
    }, arrOut, arrA, arrB);

    set_output("primOut", get_input("primOut"));
  }
};

ZENDEFNODE(PrimitiveBinaryOp,
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


struct PrimitiveMix : zeno::INode {
    virtual void apply() override{
        auto primA = get_input<PrimitiveObject>("primA");
        auto primB = get_input<PrimitiveObject>("primB");
        auto primOut = get_input<PrimitiveObject>("primOut");
        auto attrA = std::get<std::string>(get_param("attrA"));
        auto attrB = std::get<std::string>(get_param("attrB"));
        auto attrOut = std::get<std::string>(get_param("attrOut"));
        auto const &arrA = primA->attr(attrA);
        auto const &arrB = primB->attr(attrB);
        auto &arrOut = primOut->attr(attrOut);
        auto coef = get_input<zeno::NumericObject>("coef")->get<float>();
        
        std::visit([coef](auto &arrA, auto &arrB, auto &arrOut) {
          if constexpr (std::is_same_v<decltype(arrA), decltype(arrB)> && std::is_same_v<decltype(arrA), decltype(arrOut)>) {
#pragma omp parallel for
            for (int i = 0; i < arrOut.size(); i++) {
                arrOut[i] = (1.0-coef)*arrA[i] + coef*arrB[i];
            }
          }
        }, arrA, arrB, arrOut);
        set_output("primOut", get_input("primOut"));
    }
};
ZENDEFNODE(PrimitiveMix,
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

struct PrimitiveHalfBinaryOp : zeno::INode {
  virtual void apply() override {
    auto primA = get_input<PrimitiveObject>("primA");
    auto primOut = get_input<PrimitiveObject>("primOut");
    auto attrA = std::get<std::string>(get_param("attrA"));
    auto attrOut = std::get<std::string>(get_param("attrOut"));
    auto op = std::get<std::string>(get_param("op"));
    auto const &arrA = primA->attr(attrA);
    auto &arrOut = primOut->attr(attrOut);
    auto const &valB = get_input<NumericObject>("valueB")->value;
    std::visit([op](auto &arrOut, auto const &arrA, auto const &valB) {
        if constexpr (is_decay_same_v<decltype(arrOut[0]),
            zeno::is_vec_promotable_t<decltype(arrA[0]), decltype(valB)>>) {
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
            _PER_OP("pow", zeno::pow(a, b))
            _PER_OP("rpow", zeno::pow(b, a))
            _PER_OP("atan2", zeno::atan2(a, b))
            _PER_OP("ratan2", zeno::atan2(b, a))
#undef _PER_OP
            } else {
                throw zeno::Exception("Bad operator type: " + op);
            }
        } else {
            throw zeno::Exception("Failed to promote variant type");
        }
    }, arrOut, arrA, valB);

    set_output("primOut", get_input("primOut"));
  }
};

ZENDEFNODE(PrimitiveHalfBinaryOp,
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

}
