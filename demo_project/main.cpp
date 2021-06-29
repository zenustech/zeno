#include <zeno/zeno.h>
#include <cstdio>
#include <zeno/NumericObject.h>

struct Number : zeno::IObject {
    int value;
};


struct MakeNumber : zeno::INode {
    virtual void apply() override {
        printf("MakeNumber::apply() called!\n");
        int value = get_param<int>("value");
        auto obj = std::make_shared<Number>();
        obj->value = value;
        set_output("obj", std::move(obj));
    }
};

ZENDEFNODE(MakeNumber,
   { /* inputs: */ {
   }, /* outputs: */ {
       "obj",
   }, /* params: */ {
       {"int", "value", "233 0"},  // defl min max; defl min; defl
   }, /* category: */ {
       "demo_project",
   }});


struct NumberAdd : zeno::INode {
  virtual void apply() override {
      printf("NumberAdd::apply() called!\n");
      auto lhs = get_input<Number>("lhs");
      auto rhs = get_input<Number>("rhs");
      auto result = std::make_shared<Number>();
      result->value = lhs->value + rhs->value;
      set_output("result", std::move(result));
  }
};

ZENDEFNODE(NumberAdd,
    { /* inputs: */ {
        "lhs", "rhs",
    }, /* outputs: */ {
        "result",
    }, /* params: */ {
    }, /* category: */ {
    "demo_project",
    }});


struct NumberPrint : zeno::INode {
    virtual void apply() override {
        printf("NumberPrint::apply() called!\n");
        if (has_input("obj")) {
            auto obj = get_input<Number>("obj");
            printf("NumberPrint: object value is %d\n", obj->value);
        } else {
            printf("input socket `obj` not connected!\n");
        }
    }
};

ZENDEFNODE(NumberPrint,
   { /* inputs: */ {
           "obj",
   }, /* outputs: */ {
   }, /* params: */ {
   }, /* category: */ {
           "demo_project",
   }});
