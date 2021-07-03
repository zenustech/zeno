#include <zeno/zeno.h>
#include <zeno/NumericObject.h>
#include <iostream>
#include <cstdlib>

#ifdef _MSC_VER
static inline double drand48() {
	return rand() / (double)RAND_MAX;
}
#endif


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
    }
};

ZENDEFNODE(PrintNumeric, {
    {"value"},
    {},
    {{"string", "hint", "PrintNumeric"}},
    {"numeric"},
});
