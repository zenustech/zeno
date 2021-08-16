#include <zeno/zeno.h>
#include <iostream>

namespace {

struct LiterialInt : zeno::INode {
    virtual void apply() override {
        auto value = get_param<int>("value");
        set_output2("value", value);
    }
};

ZENDEFNODE(LiterialInt, {
    {{"int", "value", "0"}},
    {{"int", "value"}},
    {},
    {"numeric"},
});

struct LiterialFloat : zeno::INode {
    virtual void apply() override {
        auto value = get_param<float>("value");
        set_output2("value", value);
    }
};

ZENDEFNODE(LiterialFloat, {
    {{"float", "value", "0"}},
    {{"float", "value"}},
    {},
    {"numeric"},
});

struct PrintLiterial : zeno::INode {
    virtual void apply() override {
        auto value = get_input2<zeno::scalar_type_variant>("value");
        auto hint = get_param<std::string>("hint");
        std::cout << hint << ": ";
        std::visit([&] (auto const &val) {
            std::cout << val << std::endl;
        }, value);
    }
};

ZENDEFNODE(PrintLiterial, {
    {{"float", "value"}},
    {},
    {{"string", "hint", "PrintLiterial"}},
    {"numeric"},
});

}
