#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {

std::shared_ptr<PrimitiveObject> primitive_merge(std::shared_ptr<zeno::ListObject> list) {
    auto outprim = std::make_shared<PrimitiveObject>();

    size_t len = 0;
    size_t poly_len = 0;

    //fix pyb
    for (auto const &prim: list->get<std::shared_ptr<PrimitiveObject>>()) {
        prim->foreach_attr([&] (auto const &key, auto const &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            outprim->add_attr<T>(key);
        });
    }
    //fix pyb

    for (auto const &prim: list->get<std::shared_ptr<PrimitiveObject>>()) {
        const auto base = outprim->size();
        prim->foreach_attr([&] (auto const &key, auto const &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            //fix pyb
            auto &outarr = outprim->attr<T>(key);
            outarr.insert(outarr.end(), std::begin(arr), std::end(arr));
            //for (auto const &val: arr) outarr.push_back(val);
            //end fix pyb
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
        for (auto const &idx: prim->loops) {
            outprim->loops.push_back(idx + len);
        }
        size_t sub_poly_len = 0;
        for (auto const &poly: prim->polys) {
            sub_poly_len = std::max(sub_poly_len, (size_t)(poly.first + poly.second));
            outprim->polys.emplace_back(poly.first + poly_len, poly.second);
        }
        poly_len += sub_poly_len;
        len += prim->size();
        //fix pyb
        outprim->resize(len);
    }

    return outprim;
}

struct PrimitiveMerge : zeno::INode {
  virtual void apply() override {
    auto list = get_input<ListObject>("listPrim");
    auto outprim = primitive_merge(list);

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
