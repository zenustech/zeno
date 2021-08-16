#include <zeno/zeno.h>
#include <iostream>

namespace {

struct LiterialInt : zeno::INode {
    virtual void apply() override {
        auto value = get_input2<int>("value");
        set_output2("value", value);
    }
};

ZENDEFNODE(LiterialInt, {
    {{"int", "value", ""}},
    {{"int", "value"}},
    {},
    {"numeric"},
});

struct LiterialFloat : zeno::INode {
    virtual void apply() override {
        auto value = get_input2<float>("value");
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
        auto hint = get_input2<std::string>("hint");
        std::cout << hint << ": ";
        std::visit([&] (auto const &val) {
            std::cout << val << std::endl;
        }, value);
    }
};

ZENDEFNODE(PrintLiterial, {
    {"value", {"string", "hint", "PrintLiterial"}},
    {},
    {},
    {"numeric"},
});

}
