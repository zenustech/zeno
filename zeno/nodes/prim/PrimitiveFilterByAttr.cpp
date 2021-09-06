#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>

namespace zeno {

template <class T>
std::variant
    < std::greater<T>
    , std::less<T>
    , std::greater_equal<T>
    , std::less_equal<T>
    , std::equal_to<T>
    , std::not_equal_to<T>
> get_variant_ops(std::string const &name) {
    if (name == "cmpgt") {
        return std::greater<T>();
    } else if (name == "cmplt") {
        return std::less<T>();
    } else if (name == "cmpge") {
        return std::greater_equal<T>();
    } else if (name == "cmple") {
        return std::less_equal<T>();
    } else if (name == "cmpeq") {
        return std::equal_to<T>();
    } else if (name == "cmpne") {
        return std::not_equal_to<T>();
    }
}

struct PrimitiveFilterByAttr : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto attrName = get_param<std::string>("attrName");
    auto acceptIf = get_param<std::string>("acceptIf");
    //auto vecSelType = get_param<std::string>("vecSelType");

    auto &attr = prim->attr<float>(attrName);
    auto value = get_input2<float>("value");
    std::vector<int> revamp;
    std::visit([&] (auto op) {
        for (int i = 0; i < attr.size(); i++) {
            if (zeno::alltrue(op(attr[i], value)))
                revamp.emplace_back(i);
        }
    }, get_variant_ops<float>(acceptIf));

    prim->foreach_attr([&] (auto const &key, auto &arr) {
        for (int i = 0; i < revamp.size(); i++) {
            arr[i] = arr[revamp[i]];
        }
    });
    prim->resize(revamp.size());

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveFilterByAttr,
    { /* inputs: */ {
    "prim",
    {"NumericObject", "value", "0"},
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"string", "attrName", "rad"},
    {"enum cmpgt cmplt cmpge cmple cmpeq cmpne", "acceptIf", "cmpgt"},
    //{"enum any all", "vecSelType", "all"},
    }, /* category: */ {
    "primitive",
    }});


}
