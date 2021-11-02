#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <cstdio>

namespace {

struct NumericInterpolation : zeno::INode {
    template <class T1, class T2, class T3, class Void = void>
    struct uninterp {
        auto operator()(T1 const &src, T2 const &srcMin, T3 const &srcMax) {
            printf("ERROR: failed to uninterp, type mismatch\n");
            return 0;
        }
    };

    template <class T1, class T2, class T3>
    struct uninterp<T1, T2, T3, std::void_t<decltype(
            (std::declval<T1>() - std::declval<T2>()) * (
                1.f / (std::declval<T3>() - std::declval<T2>()))
        )>> {
        auto operator()(T1 const &src, T2 const &srcMin, T3 const &srcMax) {
            return (src - srcMin) * (1.f / (srcMax - srcMin));
        }
    };

    template <class T1, class T2, class T3, class Void = void>
    struct interp {
        auto operator()(T1 const &fac, T2 const &dstMin, T3 const &dstMax) {
            printf("ERROR: failed to interp, type mismatch\n");
            return 0;
        }
    };

    template <class T1, class T2, class T3>
    struct interp<T1, T2, T3, std::void_t<decltype(
            std::declval<T1>() * (std::declval<T3>() - std::declval<T2>()) + std::declval<T2>()
        )>> {
        auto operator()(T1 const &fac, T2 const &dstMin, T3 const &dstMax) {
            return fac * (dstMax - dstMin) + dstMin;
        }
    };

    template <class T1, class T2, class T3>
    static auto uninterp_f(T1 const &src, T2 const &srcMin, T3 const &srcMax) {
        return uninterp<T1, T2, T3>()(src, srcMin, srcMax);
    }

    template <class T1, class T2, class T3>
    static auto interp_f(T1 const &fac, T2 const &dstMin, T3 const &dstMax) {
        return interp<T1, T2, T3>()(fac, dstMin, dstMax);
    }

    virtual void apply() override {
        auto src = has_input("src") ? get_input<zeno::NumericObject>("src")->value : 0.5f;
        auto srcMin = has_input("srcMin") ? get_input<zeno::NumericObject>("srcMin")->value : 0;
        auto srcMax = has_input("srcMax") ? get_input<zeno::NumericObject>("srcMax")->value : 1;
        auto dstMin = has_input("dstMin") ? get_input<zeno::NumericObject>("dstMin")->value : 0;
        auto dstMax = has_input("dstMax") ? get_input<zeno::NumericObject>("dstMax")->value : 1;
        auto isClamped = get_param<bool>("isClamped");

        zeno::NumericValue fac;
        std::visit([&fac, isClamped] (auto src, auto srcMin, auto srcMax) {
            auto f = uninterp_f(src, srcMin, srcMax);
            if (isClamped)
                f = zeno::max(0, zeno::min(f, 1));
            fac = f;
        }, src, srcMin, srcMax);

        zeno::NumericValue dst;
        std::visit([&dst] (auto fac, auto dstMin, auto dstMax) {
            dst = interp_f(fac, dstMin, dstMax);
        }, fac, dstMin, dstMax);

        auto ret = std::make_shared<zeno::NumericObject>();
        ret->value = dst;
        set_output("dst", std::move(ret));
    }
};

ZENDEFNODE(NumericInterpolation, {
    {{"NumericObject", "src"}, {"NumericObject", "srcMin", "0"},
     {"NumericObject", "srcMax", "1"}, {"NumericObject", "dstMin", "0"},
     {"NumericObject", "dstMax", "1"}},
    {{"NumericObject", "dst"}},
    {{"bool", "isClamped", "0"}},
    {"numeric"},
});

}
