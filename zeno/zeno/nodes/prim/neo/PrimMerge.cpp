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
            bases[0] = 0;
            size_t total = 0;
            for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
                auto prim = primList[primIdx].get();
                total += prim->verts.size();
                bases[primIdx + 1] = total;
            }
            outprim->verts.resize(total);

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
