#include "zeno/types/StringObject.h"
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>

namespace zeno {

#define DEFINE_FUNCTOR(type, func) \
struct type { \
    template <class ...Ts> \
    auto operator()(Ts &&...ts) const { \
        return func(std::forward<Ts>(ts)...); \
    } \
}

#define DEFINE_FUNCTOR_UNOP(type, op) \
struct type { \
    template <class T1> \
    auto operator()(T1 &&t1) const { \
        return op std::forward<T1>(t1); \
    } \
}

#define DEFINE_FUNCTOR_BINOP(type, op) \
struct type { \
    template <class T1, class T2> \
    auto operator()(T1 &&t1, T2 &&t2) const { \
        return std::forward<T1>(t1) op std::forward<T2>(t2); \
    } \
}

DEFINE_FUNCTOR_BINOP(cmpgt_t, >);
DEFINE_FUNCTOR_BINOP(cmplt_t, <);
DEFINE_FUNCTOR_BINOP(cmpge_t, >=);
DEFINE_FUNCTOR_BINOP(cmple_t, <=);
DEFINE_FUNCTOR_BINOP(cmpeq_t, ==);
DEFINE_FUNCTOR_BINOP(cmpne_t, !=);
DEFINE_FUNCTOR(anytrue_t, zeno::anytrue);
DEFINE_FUNCTOR(alltrue_t, zeno::alltrue);

static std::variant
    < cmpgt_t
    , cmplt_t
    , cmpge_t
    , cmple_t
    , cmpeq_t
    , cmpne_t
> get_variant_ops(std::string const &name) {
    if (name == "cmpgt") {
        return cmpgt_t();
    } else if (name == "cmplt") {
        return cmplt_t();
    } else if (name == "cmpge") {
        return cmpge_t();
    } else if (name == "cmple") {
        return cmple_t();
    } else if (name == "cmpeq") {
        return cmpeq_t();
    } else if (name == "cmpne") {
        return cmpne_t();
    } else {
        throw Exception("bad compare operator: " + name);
    }
}

static std::variant
    < anytrue_t
    , alltrue_t
> get_anyall_ops(std::string const &name) {
    if (name == "any") {
        return anytrue_t();
    } else if (name == "all") {
        return alltrue_t();
    } else {
        throw Exception("bad anyall operator: " + name);
    }
}

struct PrimitiveFilterByAttr : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto attrName = get_param<std::string>("attrName");
    auto acceptIf = get_param<std::string>("acceptIf");
    auto vecSelType = get_param<std::string>("vecSelType");
    
    std::vector<int> revamp;
    prim->attr_visit(attrName, [&] (auto const &attr) {
        using T = std::decay_t<decltype(attr[0])>;
        auto value = get_input2<T>("value");
        std::visit([&] (auto op, auto aop) {
            for (int i = 0; i < attr.size(); i++) {
                if (aop(op(attr[i], value)))
                    revamp.emplace_back(i);
            }
        }
        , get_variant_ops(acceptIf)
        , get_anyall_ops(vecSelType)
        );
    });

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
    {"enum any all", "vecSelType", "all"},
    }, /* category: */ {
    "primitive",
    }});



struct SubLine : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("line");
    auto attrName = get_param<std::string>("attrName");
    auto acceptIf = get_param<std::string>("acceptIf");
    auto vecSelType = get_param<std::string>("vecSelType");
    
    std::vector<int> revamp;
    prim->attr_visit(attrName, [&] (auto const &attr) {
        using T = std::decay_t<decltype(attr[0])>;
        auto value = get_input2<T>("value");
        std::visit([&] (auto op, auto aop) {
            for (int i = 0; i < attr.size(); i++) {
                if (aop(op(attr[i], value)))
                    revamp.emplace_back(i);
            }
        }
        , get_variant_ops(acceptIf)
        , get_anyall_ops(vecSelType)
        );
    });

    prim->foreach_attr([&] (auto const &key, auto &arr) {
        for (int i = 0; i < revamp.size(); i++) {
            arr[i] = arr[revamp[i]];
        }
    });
    prim->resize(revamp.size());
    int i=0;
    for(i=0;i<prim->lines.size();i++)
    {
        if(prim->lines[i][0]>=prim->verts.size()||prim->lines[i][1]>=prim->verts.size())
            break;
    }
    prim->lines.resize(i);
    
    set_output("prim", get_input("line"));
  }
};
ZENDEFNODE(SubLine,
    { /* inputs: */ {
    "line",
    {"NumericObject", "value", "0"},
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"string", "attrName", "rad"},
    {"enum cmpgt cmplt cmpge cmple cmpeq cmpne", "acceptIf", "cmpgt"},
    {"enum any all", "vecSelType", "all"},
    }, /* category: */ {
    "primitive",
    }});



}

