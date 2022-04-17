#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/safe_dynamic_cast.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {

struct MakePrimitiveFromList : zeno::INode {
  virtual void apply() override {
    auto prim = std::make_shared<PrimitiveObject>();
    auto list = get_input<ListObject>("list");
    for (auto const &val: list->getLiterial<vec3f>()) {
        prim->verts.push_back(val);
    }
    prim->verts.update();
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(MakePrimitiveFromList,
    { /* inputs: */ {
    {"ListObject", "list"},
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});

}
