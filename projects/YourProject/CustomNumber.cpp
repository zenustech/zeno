// This is an demo on how to use custom objects in ZENO
#include <zeno/zeno.h>
#include <cstdio>

struct CustomNumber : zeno::IObject {
    int value;
};


struct MakeCustomNumber : zeno::INode {
    virtual void apply() override {
        printf("MakeCustomNumber::apply() called!\n");
        int value = get_param<int>("value");

        auto obj = std::make_shared<CustomNumber>();

        obj->value = value;
        set_output("obj", std::move(obj));
    }
};

ZENDEFNODE(
    MakeCustomNumber,
    { 
        /* inputs: */ 
        {
        }, 
        /* outputs: */ 
        {
            "obj",
        }, 
        /* params: */ 
        {
            {"int", "value", "233 0"},  
            // defl min max; defl min; defl
        }, 
        /* category: */ 
        {
            "YourProject",
        }
    }
);


struct CustomNumberAdd : zeno::INode {
    virtual void apply() override {
        printf("CustomNumberAdd::apply() called!\n");
        auto lhs = get_input<CustomNumber>("lhs");
        auto rhs = get_input<CustomNumber>("rhs");

        auto result = std::make_shared<CustomNumber>();

        result->value = lhs->value + rhs->value;
        set_output("result", std::move(result));
    }
};

ZENDEFNODE(CustomNumberAdd,
        { /* inputs: */ {
        "lhs", "rhs",
        }, /* outputs: */ {
        "result",
        }, /* params: */ {
        }, /* category: */ {
        "YourProject",
        }});


struct CustomNumberPrint : zeno::INode {
    virtual void apply() override {
        printf("CustomNumberPrint::apply() called!\n");
        if (has_input("obj")) {
            auto obj = get_input<CustomNumber>("obj");
            printf("CustomNumberPrint: object value is %d\n", obj->value);
        } else {
            printf("input socket `obj` not connected!\n");
        }
    }
};

ZENDEFNODE(
    CustomNumberPrint,
    { 
        /* inputs: */ 
        {
            "obj",
        }, 
        /* outputs: */ 
        {
        }, 
        /* params: */ 
        {
        }, 
        /* category: */ 
        {
            "YourProject",
        }
    }
);
