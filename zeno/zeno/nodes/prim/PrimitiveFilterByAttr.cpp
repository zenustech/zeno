#include "zeno/types/StringObject.h"
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>

namespace zeno {
namespace {

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
    auto valueObj = get_input<NumericObject>("value");
    
    std::vector<int> revamp;
    revamp.reserve(prim->size());
    prim->attr_visit(attrName, [&] (auto const &attr) {
        using T = std::decay_t<decltype(attr[0])>;
        auto value = valueObj->get<T>();
        std::visit([&] (auto op, auto aop) {
            int n = std::min(prim->size(), attr.size());
            for (int i = 0; i < n; i++) {
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
    auto old_prim_size = prim->size();
    prim->resize(revamp.size());

    if (get_param<bool>("mockTopos") && (0
            || prim->tris.size()
            || prim->quads.size()
            || prim->lines.size()
            || prim->polys.size()
            || prim->points.size()
         )) {

        std::vector<int> unrevamp(old_prim_size, -1);
        for (int i = 0; i < revamp.size(); i++) {
            unrevamp[revamp[i]] = i;
        }
        auto mock = [&] (int &x) -> bool {
            int loc = unrevamp[x];
            if (loc == -1)
                return false;
            x = loc;
            return true;
        };

        if (prim->tris.size()) {
            std::vector<int> trisrevamp;
            trisrevamp.reserve(prim->tris.size());
            for (int i = 0; i < prim->tris.size(); i++) {
                auto &tri = prim->tris[i];
                if (mock(tri[0]) && mock(tri[1]) && mock(tri[2]))
                    trisrevamp.emplace_back(i);
            }
            for (int i = 0; i < trisrevamp.size(); i++) {
                prim->tris[i] = prim->tris[trisrevamp[i]];
            }
            prim->tris.foreach_attr([&] (auto const &key, auto &arr) {
                for (int i = 0; i < trisrevamp.size(); i++) {
                    arr[i] = arr[trisrevamp[i]];
                }
            });
            prim->tris.resize(trisrevamp.size());
        }

        if (prim->quads.size()) {
            std::vector<int> quadsrevamp;
            quadsrevamp.reserve(prim->quads.size());
            for (int i = 0; i < prim->quads.size(); i++) {
                auto &quad = prim->quads[i];
                if (mock(quad[0]) && mock(quad[1]) && mock(quad[2]) && mock(quad[3]))
                    quadsrevamp.emplace_back(i);
            }
            for (int i = 0; i < quadsrevamp.size(); i++) {
                prim->quads[i] = prim->quads[quadsrevamp[i]];
            }
            prim->quads.foreach_attr([&] (auto const &key, auto &arr) {
                for (int i = 0; i < quadsrevamp.size(); i++) {
                    arr[i] = arr[quadsrevamp[i]];
                }
            });
            prim->quads.resize(quadsrevamp.size());
        }

        if (prim->lines.size()) {
            std::vector<int> linesrevamp;
            linesrevamp.reserve(prim->lines.size());
            for (int i = 0; i < prim->lines.size(); i++) {
                auto &line = prim->lines[i];
                if (mock(line[0]) && mock(line[1]))
                    linesrevamp.emplace_back(i);
            }
            for (int i = 0; i < linesrevamp.size(); i++) {
                prim->lines[i] = prim->lines[linesrevamp[i]];
            }
            prim->lines.foreach_attr([&] (auto const &key, auto &arr) {
                for (int i = 0; i < linesrevamp.size(); i++) {
                    arr[i] = arr[linesrevamp[i]];
                }
            });
            prim->lines.resize(linesrevamp.size());
        }

        if (prim->polys.size()) {
            std::vector<int> polysrevamp;
            polysrevamp.reserve(prim->polys.size());
            for (int i = 0; i < prim->polys.size(); i++) {
                auto &poly = prim->polys[i];
                bool succ = [&] {
                    for (int p = poly.first; p < poly.first + poly.second; p++)
                        if (!mock(prim->loops[p]))
                            return false;
                    return true;
                }();
                if (succ)
                    polysrevamp.emplace_back(i);
            }
            for (int i = 0; i < polysrevamp.size(); i++) {
                prim->polys[i] = prim->polys[polysrevamp[i]];
            }
            prim->polys.foreach_attr([&] (auto const &key, auto &arr) {
                for (int i = 0; i < polysrevamp.size(); i++) {
                    arr[i] = arr[polysrevamp[i]];
                }
            });
            prim->polys.resize(polysrevamp.size());
        }

        if (prim->points.size()) {
            std::vector<int> pointsrevamp;
            pointsrevamp.reserve(prim->points.size());
            for (int i = 0; i < prim->points.size(); i++) {
                auto &point = prim->points[i];
                if (mock(point))
                    pointsrevamp.emplace_back(i);
            }
            for (int i = 0; i < pointsrevamp.size(); i++) {
                prim->points[i] = prim->points[pointsrevamp[i]];
            }
            prim->points.foreach_attr([&] (auto const &key, auto &arr) {
                for (int i = 0; i < pointsrevamp.size(); i++) {
                    arr[i] = arr[pointsrevamp[i]];
                }
            });
            prim->points.resize(pointsrevamp.size());
        }

    }
    
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
    {"bool", "mockTopos", "1"},
    }, /* category: */ {
    "primitive",
    }});



struct SubLine : INode { // deprecated zhxx-happy-node, FilterByAttr already auto-mock lines!
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("line");
    auto attrName = get_param<std::string>("attrName");
    auto acceptIf = get_param<std::string>("acceptIf");
    auto vecSelType = get_param<std::string>("vecSelType");
    auto valueObj = get_input<NumericObject>("value");
    
    std::vector<int> revamp;
    prim->attr_visit(attrName, [&] (auto const &attr) {
        using T = std::decay_t<decltype(attr[0])>;
        auto value = valueObj->get<T>();
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
}

