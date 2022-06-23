#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <iostream>
#include <cstdlib>

namespace zeno {
namespace {

struct PrintNumeric : zeno::INode {
    template <class T>
    struct do_print {
        do_print(T const &x) {
            std::cout << x;
        }
    };

    template <size_t N, class T>
    struct do_print<zeno::vec<N, T>> {
        do_print(zeno::vec<N, T> const &x) {
            std::cout << "(";
            for (int i = 0; i < N; i++) {
                std::cout << x[i];
                if (i != N - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << ")";
        }
    };

    virtual void apply() override {
        auto obj = get_input<zeno::NumericObject>("value");
        auto hint = get_param<std::string>("hint");
        std::cout << hint << ": ";
        std::visit([](auto const &val) {
            do_print _(val);
        }, obj->value);
        std::cout << std::endl;
        set_output("value", std::move(obj));
    }
};

ZENDEFNODE(PrintNumeric, {
    {{"NumericObject", "value"}},
    {{"NumericObject", "value"}},
    {{"string", "hint", "PrintNumeric"}},
    {"numeric"},
});


//struct ToVisualize_NumericObject : PrintNumeric {
    //virtual void apply() override {
        //inputs["hint:"] = std::make_shared<zeno::StringObject>("VIEW of NumericObject");
        //PrintNumeric::apply();
    //}
//};

//ZENO_DEFOVERLOADNODE(ToVisualize, _NumericObject, typeid(zeno::NumericObject).name())({
        //{"value"},
        //{},
        //{{"string", "path", ""}},
        //{"numeric"},
//});

}
}
