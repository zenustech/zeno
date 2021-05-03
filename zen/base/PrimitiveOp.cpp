#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/NumericObject.h>
#include <cstring>
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
    auto size = std::get<int>(get_param("size"));
    prim->resize(size);
  }
};

static int defPrimitiveResize = zen::defNodeClass<PrimitiveResize>("PrimitiveResize",
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    }, /* params: */ {
    {"int", "size", "0 0"},
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
    void operator()(std::vector<TOut> &arrOut, std::vector<TA> arrA) {
        #pragma omp parallel for
        for (int i = 0; i < arrA.size(); i++) {
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
    auto &arrOut = primOut->attr(attrOut);
    auto const &arrA = primA->attr(attrA);
    std::visit([op](auto &arrOut, auto const &arrA) {
        if constexpr (std::is_same<decltype(arrOut[0]), decltype(arrA[0])>::value) {
            if (op == "copy") {
                UnaryOperator([](auto const &a) { return a; })(arrOut, arrA);
            } else if (op == "neg") {
                UnaryOperator([](auto const &a) { return -a; })(arrOut, arrA);
            } else {
                printf("%s\n", op.c_str());
                assert(0 && "Bad operator type");
            }
        }
    }, arrOut, arrA);
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
    auto arr = prim->attr(attrName);
    std::visit([](auto const &arr) {
        for (int i = 0; i < arr.size(); i++) {
            print_cout(arr[i]);
        }
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
    auto attrName = std::get<std::string>(get_param("attrName"));
    auto arr = prim->attr(attrName);
    std::visit([attrName, value](auto &arr) {
        printf("attribute %s, length %d:", attrName.c_str(), arr.size());
        for (int i = 0; i < arr.size(); i++) {
            arr[i] = decltype(arr[i])(value);
        }
        if (arr.size() == 0) {
            printf("(no data)\n");
        }
        printf("\n");
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
    }, /* category: */ {
    "primitive",
    }});

}
