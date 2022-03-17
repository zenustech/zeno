#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

struct PrimMerge : INode {
    virtual void apply() override {
        auto list = get_input<ListObject>("listPrim");
        auto primList = list->get<std::shared_ptr<PrimitiveObject>>();
        auto tagAttr = get_input<StringObject>("tagAttr")->get();
        auto outprim = std::make_shared<PrimitiveObject>();

        if (primList.size()) {
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
                auto prim = primList[primIdx].get();
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
                auto prim = primList[primIdx].get();
                auto base = bases[primIdx];
                auto core = [&] (auto key, auto const &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
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
                };
                core(std::true_type{}, prim->verts.values);
                prim->verts.foreach_attr(core);
            }

            for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
                auto prim = primList[primIdx].get();
                auto vbase = bases[primIdx];
                auto base = linebases[primIdx];
                auto core = [&] (auto key, auto const &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
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
                };
                core(std::true_type{}, prim->lines.values);
                prim->lines.foreach_attr(core);
            }

            for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
                auto prim = primList[primIdx].get();
                auto vbase = bases[primIdx];
                auto base = linebases[primIdx];
                auto core = [&] (auto key, auto const &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
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
                };
                core(std::true_type{}, prim->tris.values);
                prim->tris.foreach_attr(core);
            }

            for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
                auto prim = primList[primIdx].get();
                auto vbase = bases[primIdx];
                auto base = linebases[primIdx];
                auto core = [&] (auto key, auto const &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
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
                };
                core(std::true_type{}, prim->quads.values);
                prim->quads.foreach_attr(core);
            }
        }

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
