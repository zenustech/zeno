#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveTools.h>

namespace zeno {

struct LineAddVert : zeno::INode {//make zhxx happy
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto vert = get_input<PrimitiveObject>("vert");
    for(auto key:vert->attr_keys())
            {
                if (key != "pos")
                std::visit([&prim, key](auto &&ref) {
                    using T = std::remove_cv_t<std::remove_reference_t<decltype(ref[0])>>;
                    //printf("key: %s, T: %s\n", key.c_str(), typeid(T).name());
                    prim->add_attr<T>(key);
                }, vert->attr(key));
                // using T = std::decay_t<decltype(refprim->attr(key)[0])>;
                // outprim->add_attr<T>(key);
            }
    addIndividualPrimitive(prim.get(), vert.get(), 0);
    prim->lines.push_back(zeno::vec2i(prim->verts.size()-2, prim->verts.size()-1));
    //set_output("prim", std::move(prim));
  }
};
ZENDEFNODE(LineAddVert,
    { /* inputs: */ {
    "prim", "vert",
    }, /* outputs: */ {
    //"prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});

}
