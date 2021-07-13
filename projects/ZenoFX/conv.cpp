#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include "Particles.h"
#include <type_traits>

struct ParticlesFromPrimitive : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto pars = std::make_shared<Particles>();

        for (auto const &[key, attr]: prim->m_attrs) {
            std::visit([&] (auto const &arr) {
                using T0 = std::decay_t<decltype(arr[0])>;
                constexpr size_t N =
                    zeno::is_vec_v<T0> ? zeno::is_vec_n<T0> : 1;
                using T = zeno::decay_vec_t<T0>;
                auto &outarr = pars->add_attr<T, N>(key);
                outarr.resize(arr.size());
            }, attr);
        }

        set_output("pars", std::move(pars));
    }
};

ZENDEFNODE(ParticlesFromPrimitive, {
    {"prim"},
    {"pars"},
    {},
    {"zenofx"},
});
