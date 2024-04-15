#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveTools.h>

namespace zeno {
namespace {

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
    "deprecated",
    }});

struct SyncPrimitiveAttributes : zeno::INode {
    virtual void apply() override {
        auto prim1 = get_input<zeno::PrimitiveObject>("prim1");
        auto prim2 = get_input<zeno::PrimitiveObject>("prim2");

        prim1->verts.foreach_attr([&] (auto const &key, auto const &attr) {
            using T = std::decay_t<decltype(attr[0])>;
            prim2->add_attr<T>(key);
        });

        prim2->verts.foreach_attr([&] (auto const &key, auto const &attr) {
            using T = std::decay_t<decltype(attr[0])>;
            prim1->add_attr<T>(key);
        });

        // prim1->resize(prim1->size());
        // prim2->resize(prim2->size());

        set_output("prim1",prim1);
        set_output("prim2",prim2);
    }
};

ZENDEFNODE(SyncPrimitiveAttributes, {
    {"prim1", "prim2"},
    {"prim1", "prim2"},
    {},
    {"primitive"},
});

}
}
