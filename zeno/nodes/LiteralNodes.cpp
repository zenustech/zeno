#include <zeno/zeno.h>
#include <iostream>

namespace {

struct LiteralInt : zeno::INode {
    virtual void apply() override {
        auto value = get_input2<int>("value");
        set_output2("value", value);
    }
};

ZENDEFNODE(LiteralInt, {
    {{"int", "value", "0"}},
    {{"int", "value"}},
    {},
    {"literal"},
});

struct LiteralFloat : zeno::INode {
    virtual void apply() override {
        auto value = get_input2<float>("value");
        set_output2("value", value);
    }
};

ZENDEFNODE(LiteralFloat, {
    {{"float", "value", "0"}},
    {{"float", "value"}},
    {},
    {"literal"},
});

struct PrintLiteral : zeno::INode {
    virtual void apply() override {
        auto value = get_input2<zeno::scalar_type_variant>("value");
        auto hint = get_input2<std::string>("hint");
        std::cout << hint << ": ";
        std::visit([&] (auto const &val) {
            std::cout << val << std::endl;
        }, value);
    }
};

ZENDEFNODE(PrintLiteral, {
    {"value", {"string", "hint", "PrintLiteral"}},
    {},
    {},
    {"literal"},
});

}
