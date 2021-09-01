#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {


struct PrimitiveMerge : zeno::INode {
  virtual void apply() override {
    auto list = get_input<ListObject>("listPrim");
    auto outprim = std::make_shared<PrimitiveObject>();

    size_t len = 0;
    for (auto const &prim: list->get<std::shared_ptr<PrimitiveObject>>()) {
        prim->foreach_attr([&] (auto const &key, auto const &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            auto &outarr = outprim->add_attr<T>(key);
            for (auto const &val: arr) {
                outarr.push_back(val);
            }
        });
        for (auto const &idx: prim->points) {
            outprim->points.push_back(idx + len);
        }
        for (auto const &idx: prim->lines) {
            outprim->lines.push_back(idx + len);
        }
        for (auto const &idx: prim->tris) {
            outprim->tris.push_back(idx + len);
        }
        for (auto const &idx: prim->quads) {
            outprim->quads.push_back(idx + len);
        }
        len += prim->size();
    }
    outprim->resize(len);

    set_output("prim", std::move(outprim));
  }
};

ZENDEFNODE(PrimitiveMerge, {
    {"listPrim"},
    {"prim"},
    {},
    {"primitive"},
});


}
