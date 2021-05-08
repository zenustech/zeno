#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/NumericObject.h>
#include <glm/glm.hpp>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zenbase {

struct MakePrimitive : zen::INode {
  virtual void apply() override {
    auto prim = zen::IObject::make<PrimitiveObject>();
    set_output("prim", prim);
  }
};

static int defMakePrimitive = zen::defNodeClass<MakePrimitive>("MakePrimitive",
    { /* inputs: */ {
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveResize : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto size = get_input("size")->as<NumericObject>()->get<int>();
    prim->resize(size);
  }
};

static int defPrimitiveResize = zen::defNodeClass<PrimitiveResize>("PrimitiveResize",
    { /* inputs: */ {
    "prim",
    "size",
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveAddAttr : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto name = std::get<std::string>(get_param("name"));
    auto type = std::get<std::string>(get_param("type"));
    if (type == "float") {
        prim->add_attr<float>(name);
    } else if (type == "float3") {
        prim->add_attr<glm::vec3>(name);
    } else {
        printf("%s\n", type.c_str());
        assert(0 && "Bad attribute type");
    }
  }
};

static int defPrimitiveAddAttr = zen::defNodeClass<PrimitiveAddAttr>("PrimitiveAddAttr",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    }, /* params: */ {
    {"string", "name", "pos"},
    {"string", "type", "float3"},
    }, /* category: */ {
    "primitive",
    }});


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
            arrOut[i] = decltype(arrOut[i])(val);
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
        if (0) {
#define _PER_OP(opname, expr) \
        } else if (op == opname) { \
            UnaryOperator([](auto const &a) { return expr; })(arrOut, arrA);
        _PER_OP("copy", a)
        _PER_OP("neg", -a)
        _PER_OP("sqrt", glm::sqrt(a))
        _PER_OP("sin", glm::sin(a))
        _PER_OP("cos", glm::cos(a))
        _PER_OP("tan", glm::tan(a))
        _PER_OP("asin", glm::asin(a))
        _PER_OP("acos", glm::acos(a))
        _PER_OP("atan", glm::atan(a))
        _PER_OP("exp", glm::exp(a))
        _PER_OP("log", glm::log(a))
#undef _PER_OP
        } else {
            printf("%s\n", op.c_str());
            assert(0 && "Bad operator type");
        }
    }, arrOut, arrA);
  }
};


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
            arrOut[i] = decltype(arrOut[i])(val);
        }
    }
};

static int defPrimitiveUnaryOp = zen::defNodeClass<PrimitiveUnaryOp>("PrimitiveUnaryOp",
    { /* inputs: */ {
    "primA",
    "primOut",
    }, /* outputs: */ {
    }, /* params: */ {
    {"string", "attrA", "pos"},
    {"string", "attrOut", "pos"},
    {"string", "op", "copy"},
    }, /* category: */ {
    "primitive",
    }});


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
        _PER_OP("mul", a * b)
        _PER_OP("div", a / b)
        _PER_OP("pow", glm::pow(a, b))
        _PER_OP("atan2", glm::atan(a, b))
#undef _PER_OP
        } else {
            printf("%s\n", op.c_str());
            assert(0 && "Bad operator type");
        }
    }, arrOut, arrA, arrB);
  }
};

static int defPrimitiveBinaryOp = zen::defNodeClass<PrimitiveBinaryOp>("PrimitiveBinaryOp",
    { /* inputs: */ {
    "primA",
    "primB",
    "primOut",
    }, /* outputs: */ {
    }, /* params: */ {
    {"string", "attrA", "pos"},
    {"string", "attrN", "pos"},
    {"string", "attrOut", "pos"},
    {"string", "op", "copy"},
    }, /* category: */ {
    "primitive",
    }});


void print_cout(float x) {
    printf("%f\n", x);
}

void print_cout(glm::vec3 const &a) {
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
  }
};

static int defPrimitivePrintAttr = zen::defNodeClass<PrimitivePrintAttr>("PrimitivePrintAttr",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    }, /* params: */ {
    {"string", "attrName", "pos"},
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveFillAttr : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto value = std::get<float>(get_param("value"));
    auto valueY = std::get<float>(get_param("valueY"));
    auto valueZ = std::get<float>(get_param("valueZ"));
    auto attrName = std::get<std::string>(get_param("attrName"));
    auto &arr = prim->attr(attrName);
    std::visit([value, valueY, valueZ](auto &arr) {
        #pragma omp parallel for
        for (int i = 0; i < arr.size(); i++) {
            if constexpr (std::is_same<decltype(arr[i]), glm::vec3>::value) {
                arr[i] = glm::vec3(value, valueY, valueZ);
            } else {
                arr[i] = decltype(arr[i])(value);
            }
        }
    }, arr);
  }
};

static int defPrimitiveFillAttr = zen::defNodeClass<PrimitiveFillAttr>("PrimitiveFillAttr",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    }, /* params: */ {
    {"string", "attrName", "pos"},
    {"float", "value", "0"},
    {"float", "valueY", "0"},
    {"float", "valueZ", "0"},
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
            if constexpr (std::is_same<std::decay_t<decltype(arr[i])>, glm::vec3>::value) {
                arr[i] = glm::mix(glm::vec3(min, minY, minZ), glm::vec3(max, maxY, maxZ),
                        glm::vec3(drand48(), drand48(), drand48()));
            } else {
                arr[i] = glm::mix(min, max, (float)drand48());
            }
        }
    }, arr);
  }
};

static int defPrimitiveRandomizeAttr = zen::defNodeClass<PrimitiveRandomizeAttr>("PrimitiveRandomizeAttr",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
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

}
