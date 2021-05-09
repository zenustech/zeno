#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/NumericObject.h>
#include <Hg/MathUtils.h>
#include <Hg/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zenbase {


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
        if constexpr (hg::is_castable_v<decltype(arrOut[0]), decltype(arrA[0])>) {
            if (0) {
#define _PER_OP(opname, expr) \
            } else if (op == opname) { \
                UnaryOperator([](auto const &a) { return expr; })(arrOut, arrA);
            _PER_OP("copy", a)
            _PER_OP("neg", -a)
            _PER_OP("sqrt", hg::sqrt(a))
            _PER_OP("sin", hg::sin(a))
            _PER_OP("cos", hg::cos(a))
            _PER_OP("tan", hg::tan(a))
            _PER_OP("asin", hg::asin(a))
            _PER_OP("acos", hg::acos(a))
            _PER_OP("atan", hg::atan(a))
            _PER_OP("exp", hg::exp(a))
            _PER_OP("log", hg::log(a))
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
        if constexpr (hg::is_decay_same_v<decltype(arrOut[0]),
            hg::is_promotable_t<decltype(arrA[0]), decltype(arrB[0])>>) {
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
            _PER_OP("pow", hg::pow(a, b))
            _PER_OP("rpow", hg::pow(b, a))
            _PER_OP("atan2", hg::atan2(a, b))
            _PER_OP("ratan2", hg::atan2(b, a))
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
    auto const &valueB = get_input("valueB")->as<NumericObject>()->value;
    std::visit([op](auto &arrOut, auto const &arrA, auto const &valueB) {
        auto valB = hg::tovec(valueB);
        if constexpr (hg::is_decay_same_v<decltype(arrOut[0]),
            hg::is_promotable_t<decltype(arrA[0]), decltype(valB)>>) {
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
            _PER_OP("pow", hg::pow(a, b))
            _PER_OP("rpow", hg::pow(b, a))
            _PER_OP("atan2", hg::atan2(a, b))
            _PER_OP("ratan2", hg::atan2(b, a))
#undef _PER_OP
            } else {
                printf("%s\n", op.c_str());
                assert(0 && "Bad operator type");
            }
        } else {
            printf("FUCKERASDKLJLASDHJJKHLSD %s %s %s %d\n",
            hg::typenameof<decltype(arrA[0])>(),
            hg::typenameof<decltype(valB)>(),
            hg::typenameof<hg::is_promotable_t<decltype(arrA[0]), decltype(valB)>>(),
            hg::is_promotable_v<hg::vec<3, float>, float>,
            0);
            exit(-1);
            assert(0 && "Failed to promote variant type");
        }
    }, arrOut, arrA, valueB);

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
        if constexpr (hg::is_castable_v<decltype(arr[0]), decltype(value)>) {
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
        #pragma omp parallel for
        for (int i = 0; i < arr.size(); i++) {
            if constexpr (hg::is_decay_same_v<decltype(arr[i]), hg::vec3f>) {
                arr[i] = hg::mix(hg::vec3f(min, minY, minZ), hg::vec3f(max, maxY, maxZ),
                        hg::vec3f(drand48(), drand48(), drand48()));
            } else {
                arr[i] = hg::mix(min, max, (float)drand48());
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

void print_cout(hg::vec3f const &a) {
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
