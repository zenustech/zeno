#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/para/parallel_for.h>
#include <cmath>

namespace zeno {
namespace {

struct PrimFillAttr : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto value = get_input<NumericObject>("value");
        auto attr = get_input2<std::string>("attr");
        auto type = get_input2<std::string>("type");
        std::visit([&] (auto ty) {
            using T = decltype(ty);
            auto &arr = prim->verts.add_attr<T>(attr);
            auto val = value->get<T>();
            for (size_t i = 0; i < arr.size(); i++) {
                arr[i] = val;
            }
        }, enum_variant<std::variant<
            float, vec3f, int
        >>(array_index({
            "float", "vec3f", "int"
        }, type)));
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimFillAttr, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "attr", "rad"},
    {"enum float vec3f int", "type", "float"},
    {"float", "value", "0"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

struct PrimFillColor : PrimFillAttr {
    virtual void apply() override {
        this->inputs["attr"] = std::make_shared<StringObject>("clr");
        this->inputs["type"] = std::make_shared<StringObject>("vec3f");
        PrimFillAttr::apply();
    }
};

ZENDEFNODE(PrimFillColor, {
    {
    {"PrimitiveObject", "prim"},
    {"vec3f", "value", "1,0.5,0.5"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

struct PrimFloatAttrToInt : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto attr = get_input2<std::string>("attr");
        auto attrOut = get_input2<std::string>("attrOut");
        if(prim->verts.has_attr(attr)){
            auto &inArr = prim->verts.attr<float>(attr);
            auto factor = get_input2<float>("divisor");
            if (attrOut == attr) {
                std::vector<int> outArr(inArr.size());
                parallel_for(inArr.size(), [&] (size_t i) {
                    outArr[i] = std::rint(inArr[i] * factor);
                });
                prim->verts.attrs.erase(attrOut);
                prim->verts.add_attr<int>(attrOut) = std::move(outArr);
            } else {
                auto &outArr = prim->verts.add_attr<int>(attrOut);
                parallel_for(inArr.size(), [&] (size_t i) {
                    outArr[i] = std::rint(inArr[i] * factor);
                });
            }
        }
        if(prim->tris.has_attr(attr)){
            auto &inArr = prim->tris.attr<float>(attr);
            auto factor = get_input2<float>("divisor");
            if (attrOut == attr) {
                std::vector<int> outArr(inArr.size());
                parallel_for(inArr.size(), [&] (size_t i) {
                    outArr[i] = std::rint(inArr[i] * factor);
                });
                prim->tris.attrs.erase(attrOut);
                prim->tris.add_attr<int>(attrOut) = std::move(outArr);
            } else {
                auto &outArr = prim->tris.add_attr<int>(attrOut);
                parallel_for(inArr.size(), [&] (size_t i) {
                    outArr[i] = std::rint(inArr[i] * factor);
                });
            }
        }
        if(prim->polys.has_attr(attr)){
            auto &inArr = prim->polys.attr<float>(attr);
            auto factor = get_input2<float>("divisor");
            if (attrOut == attr) {
                std::vector<int> outArr(inArr.size());
                parallel_for(inArr.size(), [&] (size_t i) {
                    outArr[i] = std::rint(inArr[i] * factor);
                });
                prim->polys.attrs.erase(attrOut);
                prim->polys.add_attr<int>(attrOut) = std::move(outArr);
            } else {
                auto &outArr = prim->polys.add_attr<int>(attrOut);
                parallel_for(inArr.size(), [&] (size_t i) {
                    outArr[i] = std::rint(inArr[i] * factor);
                });
            }
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimFloatAttrToInt, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "attr", "tag"},
    {"string", "attrOut", "tag"},
    {"float", "divisor", "1"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

struct PrimIntAttrToFloat : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto attr = get_input2<std::string>("attr");
        auto attrOut = get_input2<std::string>("attrOut");


        if(prim->verts.has_attr(attr)){
            auto &inArr = prim->verts.attr<int>(attr);
            auto factor = get_input2<float>("divisor");
            if (factor) factor = 1.0f / factor;
            if (attrOut == attr) {
                std::vector<float> outArr(inArr.size());
                parallel_for(inArr.size(), [&] (size_t i) {
                    outArr[i] = float(inArr[i]) * factor;
                });
                prim->verts.attrs.erase(attrOut);
                prim->verts.add_attr<float>(attrOut) = std::move(outArr);
            } else {
                auto &outArr = prim->verts.add_attr<float>(attrOut);
                parallel_for(inArr.size(), [&] (size_t i) {
                    outArr[i] = float(inArr[i]) * factor;
                });
            }
        }
        if(prim->tris.has_attr(attr)){
            auto &inArr = prim->tris.attr<int>(attr);
            auto factor = get_input2<float>("divisor");
            if (factor) factor = 1.0f / factor;
            if (attrOut == attr) {
                std::vector<float> outArr(inArr.size());
                parallel_for(inArr.size(), [&] (size_t i) {
                    outArr[i] = float(inArr[i]) * factor;
                });
                prim->tris.attrs.erase(attrOut);
                prim->tris.add_attr<float>(attrOut) = std::move(outArr);
            } else {
                auto &outArr = prim->tris.add_attr<float>(attrOut);
                parallel_for(inArr.size(), [&] (size_t i) {
                    outArr[i] = float(inArr[i]) * factor;
                });
            }
        }
        if(prim->polys.has_attr(attr)){
            auto &inArr = prim->polys.attr<int>(attr);
            auto factor = get_input2<float>("divisor");
            if (factor) factor = 1.0f / factor;
            if (attrOut == attr) {
                std::vector<float> outArr(inArr.size());
                parallel_for(inArr.size(), [&] (size_t i) {
                    outArr[i] = float(inArr[i]) * factor;
                });
                prim->polys.attrs.erase(attrOut);
                prim->polys.add_attr<float>(attrOut) = std::move(outArr);
            } else {
                auto &outArr = prim->polys.add_attr<float>(attrOut);
                parallel_for(inArr.size(), [&] (size_t i) {
                    outArr[i] = float(inArr[i]) * factor;
                });
            }
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimIntAttrToFloat, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "attr", "tag"},
    {"string", "attrOut", "tag"},
    {"float", "divisor", "1"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

struct PrimAttrInterp : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto prim2 = get_input<PrimitiveObject>("prim2");
        auto attr = get_input2<std::string>("attr");
        auto factor = get_input2<float>("factor");
        auto facAttr = get_input2<std::string>("facAttr");
        auto facAcc = functor_variant(facAttr.empty() ? 1 : 0,
                                      [&, &facAttr = facAttr] {
                                          auto &facArr = prim->verts.attr<float>(facAttr);
                                          return [&] (size_t i) {
                                              return facArr[i];
                                          };
                                      },
                                      [&] {
                                          return [&] (size_t i) {
                                              return factor;
                                          };
                                      });
        auto process = [&] (std::string const &key, auto &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            auto &arr2 = prim2->add_attr<T>(key);
            std::visit([&] (auto const &facAcc) {
                parallel_for(std::min(arr.size(), arr2.size()), [&] (size_t i) {
                    arr[i] = (T)mix(arr[i], arr2[i], facAcc(i));
                });
            }, facAcc);
        };
        if (attr.empty()) {
            prim->foreach_attr<AttrAcceptAll>([&] (auto const &key, auto &arr) {
                if (!facAttr.empty() && key == facAttr)
                    return;
                process(key, arr);
            });
        } else {
            prim->attr_visit<AttrAcceptAll>(attr, [&] (auto &arr) {
                process(attr, arr);
            });
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimAttrInterp, {
    {
    {"PrimitiveObject", "prim"},
    {"PrimitiveObject", "prim2"},
    {"string", "attr", ""},
    {"float", "factor", "0.5"},
    {"string", "facAttr", ""},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}
}
