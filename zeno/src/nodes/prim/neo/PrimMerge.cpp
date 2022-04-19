#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

ZENO_API std::shared_ptr<zeno::PrimitiveObject> primMerge(
        std::vector<zeno::PrimitiveObject *> const &primList,
        std::string const &tagAttr) {
    auto outprim = std::make_shared<PrimitiveObject>();

    if (!primList.size()) {
        std::vector<size_t> bases(primList.size() + 1);
        std::vector<size_t> pointbases(primList.size() + 1);
        std::vector<size_t> linebases(primList.size() + 1);
        std::vector<size_t> tribases(primList.size() + 1);
        std::vector<size_t> quadbases(primList.size() + 1);
        std::vector<size_t> loopbases(primList.size() + 1);
        std::vector<size_t> polybases(primList.size() + 1);
        size_t total = 0;
        size_t pointtotal = 0;
        size_t linetotal = 0;
        size_t tritotal = 0;
        size_t quadtotal = 0;
        size_t looptotal = 0;
        size_t polytotal = 0;
        for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
            auto prim = primList[primIdx];
            total += prim->verts.size();
            linetotal += prim->lines.size();
            tritotal += prim->tris.size();
            quadtotal += prim->quads.size();
            looptotal += prim->loops.size();
            polytotal += prim->polys.size();
            bases[primIdx + 1] = total;
            linebases[primIdx + 1] = linetotal;
            tribases[primIdx + 1] = tritotal;
            quadbases[primIdx + 1] = quadtotal;
            loopbases[primIdx + 1] = looptotal;
            polybases[primIdx + 1] = polytotal;
        }
        outprim->verts.resize(total);
        outprim->lines.resize(linetotal);
        outprim->tris.resize(tritotal);
        outprim->quads.resize(quadtotal);
        outprim->loops.resize(looptotal);
        outprim->polys.resize(polytotal);

        for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
            auto prim = primList[primIdx];
            auto base = bases[primIdx];
            auto core = [&] (auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->verts.values;
                    } else {
                        return outprim->verts.add_attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->verts.size());
                for (size_t i = 0; i < n; i++) {
                    outarr[base + i] = arr[i];
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->verts.values;
                    size_t n = std::min(arr.size(), prim->verts.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                } else {
                    auto &outarr = outprim->verts.add_attr<T>(key);
                    size_t n = std::min(arr.size(), prim->verts.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->verts.values);
            prim->verts.foreach_attr(core);
            if (tagAttr.size()) {
                auto &outarr = outprim->verts.add_attr<int>(tagAttr);
                for (size_t i = 0; i < prim->verts.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        }

        for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
            auto prim = primList[primIdx];
            auto vbase = bases[primIdx];
            auto base = linebases[primIdx];
            auto core = [&] (auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->lines.values;
                    } else {
                        return outprim->lines.add_attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->lines.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = vbase + arr[i];
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->lines.values;
                    size_t n = std::min(arr.size(), prim->lines.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = vbase + arr[i];
                    }
                } else {
                    auto &outarr = outprim->lines.add_attr<T>(key);
                    size_t n = std::min(arr.size(), prim->lines.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->lines.values);
            prim->lines.foreach_attr(core);
            if (tagAttr.size()) {
                auto &outarr = outprim->lines.add_attr<int>(tagAttr);
                for (size_t i = 0; i < prim->lines.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        }

        for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
            auto prim = primList[primIdx];
            auto vbase = bases[primIdx];
            auto base = tribases[primIdx];
            auto core = [&] (auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->tris.values;
                    } else {
                        return outprim->tris.add_attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->tris.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = vbase + arr[i];
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->tris.values;
                    size_t n = std::min(arr.size(), prim->tris.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = vbase + arr[i];
                    }
                } else {
                    auto &outarr = outprim->tris.add_attr<T>(key);
                    size_t n = std::min(arr.size(), prim->tris.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->tris.values);
            prim->tris.foreach_attr(core);
            if (tagAttr.size()) {
                auto &outarr = outprim->tris.add_attr<int>(tagAttr);
                for (size_t i = 0; i < prim->tris.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        }

        for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
            auto prim = primList[primIdx];
            auto vbase = bases[primIdx];
            auto base = quadbases[primIdx];
            auto core = [&] (auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->quads.values;
                    } else {
                        return outprim->quads.add_attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->quads.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = vbase + arr[i];
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->quads.values;
                    size_t n = std::min(arr.size(), prim->quads.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = vbase + arr[i];
                    }
                } else {
                    auto &outarr = outprim->quads.add_attr<T>(key);
                    size_t n = std::min(arr.size(), prim->quads.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->quads.values);
            prim->quads.foreach_attr(core);
            if (tagAttr.size()) {
                auto &outarr = outprim->quads.add_attr<int>(tagAttr);
                for (size_t i = 0; i < prim->quads.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        }

        for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
            auto prim = primList[primIdx];
            auto vbase = bases[primIdx];
            auto base = loopbases[primIdx];
            auto core = [&] (auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->loops.values;
                    } else {
                        return outprim->loops.add_attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->loops.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = vbase + arr[i];
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->loops.values;
                    size_t n = std::min(arr.size(), prim->loops.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = vbase + arr[i];
                    }
                } else {
                    auto &outarr = outprim->loops.add_attr<T>(key);
                    size_t n = std::min(arr.size(), prim->loops.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->loops.values);
            prim->loops.foreach_attr(core);
            if (tagAttr.size()) {
                auto &outarr = outprim->loops.add_attr<int>(tagAttr);
                for (size_t i = 0; i < prim->loops.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        }

        for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
            auto prim = primList[primIdx];
            auto lbase = loopbases[primIdx];
            auto base = polybases[primIdx];
            auto core = [&] (auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->polys.values;
                    } else {
                        return outprim->polys.add_attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->polys.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = {arr[i].first + lbase, arr[i].second};
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->polys.values;
                    size_t n = std::min(arr.size(), prim->polys.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = {arr[i].first + lbase, arr[i].second};
                    }
                } else {
                    auto &outarr = outprim->polys.add_attr<T>(key);
                    size_t n = std::min(arr.size(), prim->polys.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->polys.values);
            prim->polys.foreach_attr(core);
            if (tagAttr.size()) {
                auto &outarr = outprim->polys.add_attr<int>(tagAttr);
                for (size_t i = 0; i < prim->polys.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        }
    }

    return outprim;
}

namespace {

struct PrimMerge : INode {
    virtual void apply() override {
        auto primList = get_input<ListObject>("listPrim")->getRaw<PrimitiveObject>();
        auto tagAttr = get_input<StringObject>("tagAttr")->get();

        auto outprim = primMerge(primList, tagAttr);

        set_output("prim", std::move(outprim));
    }
};

ZENDEFNODE(PrimMerge, {
    {
        {"list", "listPrim"},
        {"string", "tagAttr", ""},
    },
    {
        {"primitive", "prim"},
    },
    {
    },
    {"primitive"},
});

}
}
