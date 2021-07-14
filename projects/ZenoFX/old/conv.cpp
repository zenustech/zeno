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

        pars->resize(prim->size());
        for (auto const &[key, attr]: prim->m_attrs) {
            std::visit([&] (auto const &arr) {
                using T0 = std::decay_t<decltype(arr[0])>;
                constexpr size_t N =
                    zeno::is_vec_v<T0> ? zeno::is_vec_n<T0> : 1;
                using T = zeno::decay_vec_t<T0>;
                auto &outarr = pars->add_attr<T, N>(key);
                outarr.resize(arr.size());
                for (int i = 0; i < arr.size(); i++) {
                    if constexpr (outarr.Dimension == 1) {
                        outarr.at(i, 0) = arr[i];
                    } else {
                        for (int j = 0; j < outarr.Dimension; j++) {
                            outarr.at(i, j) = arr[i][j];
                        }
                    }
                }
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

struct ParticlesToPrimitive : zeno::INode {
    virtual void apply() override {
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto pars = get_input<Particles>("pars");

        prim->resize(pars->size());
        pars->foreach_attr([&] (auto const &key, auto &arr) {
            if constexpr (arr.Dimension == 3) {
                auto &outarr = prim->add_attr<zeno::vec3f>(key);
                outarr.resize(arr.size());
                for (int i = 0; i < arr.size(); i++) {
                    outarr[i] = zeno::vec3f(
                        arr.at(i, 0), arr.at(i, 1), arr.at(i, 2));
                }
            } else if constexpr (arr.Dimension == 1) {
                auto &outarr = prim->add_attr<float>(key);
                outarr.resize(arr.size());
                for (int i = 0; i < arr.size(); i++) {
                    outarr[i] = arr.at(i, 0);
                }
            }
        });

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ParticlesToPrimitive, {
    {"pars"},
    {"prim"},
    {},
    {"zenofx"},
});
