#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/parallel_reduce.h>
#include <zeno/types/CurveObject.h>
#include <stdexcept>
#include <cmath>

namespace zeno {
namespace {

struct PrimFillAttr : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto value = get_input<NumericObject>("value");
        auto attr = get_input2<std::string>("attr");
        auto type = get_input2<std::string>("type");
        auto scope = get_input2<std::string>("scope");
        std::visit([&] (auto ty) {
            using T = decltype(ty);
            auto val = value->get<T>();
            if (scope == "vert") {
            auto &arr = prim->verts.add_attr<T>(attr);
                std::fill(arr.begin(), arr.end(), val);
            }
            else if (scope == "tri") {
                auto &arr = prim->tris.add_attr<T>(attr);
                std::fill(arr.begin(), arr.end(), val);
            }
            else if (scope == "loop") {
                auto &arr = prim->loops.add_attr<T>(attr);
                std::fill(arr.begin(), arr.end(), val);
            }
            else if (scope == "poly") {
                auto &arr = prim->polys.add_attr<T>(attr);
                std::fill(arr.begin(), arr.end(), val);
            }
            else if (scope == "line") {
                auto &arr = prim->lines.add_attr<T>(attr);
                std::fill(arr.begin(), arr.end(), val);
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
    {"PrimitiveObject", "prim", "", zeno::Socket_ReadOnly},
        {"enum vert tri loop poly line", "scope", "vert"},
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
        set_primitive_input("attr", "clr");
        set_primitive_input("type", "vec3f");
        set_primitive_input("scope", "vert");
        PrimFillAttr::apply();
    }
};

ZENDEFNODE(PrimFillColor, {
    {
    {"PrimitiveObject", "prim", "", zeno::Socket_ReadOnly},
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
    {"PrimitiveObject", "prim", "", zeno::Socket_ReadOnly},
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
    {"PrimitiveObject", "prim", "", zeno::Socket_ReadOnly},
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
    {"PrimitiveObject", "prim", "", zeno::Socket_ReadOnly},
    {"PrimitiveObject", "prim2", "", zeno::Socket_ReadOnly},
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

template <class T>
static void prim_remap(std::vector<T> &arr, bool autocompute, float inMax, float inMin, float outputMax, float outputMin, bool clampMax, bool clampMin, bool ramp, CurveObject *curve)
{
        if (autocompute) {
            inMin = zeno::parallel_reduce_array<T>(arr.size(), arr[0], [&] (size_t i) -> T { return arr[i]; },
            [&] (T i, T j) -> T { return zeno::min(i, j); });
            inMax = zeno::parallel_reduce_array<T>(arr.size(), arr[0], [&] (size_t i) -> T { return arr[i]; },
            [&] (T i, T j) -> T { return zeno::max(i, j); });
        }
        float val = 0.0;
        float denom = inMax - inMin;
        if(denom == 0.0) {
            parallel_for(arr.size(), [&] (size_t i) {
                val = (arr[i] < inMin ? 0. : 1.);
                if (ramp) val = curve->eval(val);
                arr[i] = val * (outputMax - outputMin) + outputMin;
            });
        }
        else if constexpr (std::is_same_v<T, float>) {
            parallel_for(arr.size(), [&] (size_t i) {
                if(clampMax) arr[i] = zeno::min(arr[i], inMax);
                if(clampMin) arr[i] = zeno::max(arr[i], inMin);
                arr[i] = (arr[i] - inMin) / (denom);
                if (ramp) arr[i] = curve->eval(arr[i]);
                arr[i] = arr[i] * (outputMax - outputMin) + outputMin;
            });
        } else if constexpr (std::is_same_v<T, int>) {
            parallel_for(arr.size(), [&] (size_t i) {
                if(clampMax) arr[i] = zeno::min(arr[i], inMax);
                if(clampMin) arr[i] = zeno::max(arr[i], inMin);
                val = (arr[i] - inMin) / (denom);
                if (ramp) val = curve->eval(val);
                arr[i] = (int) std::round(val * (outputMax - outputMin) + outputMin);
            });
        }
}


struct PrimAttrRemap : INode {
    virtual void apply() override {// change attr name to create new attr?
        auto prim = get_input<PrimitiveObject>("prim");
        auto attr = get_input2<std::string>("attr");
        auto scope = get_input2<std::string>("scope");
        auto autoCompute = get_input2<bool>("Auto Compute input range");
        auto inMin = get_input2<float>("Input min");
        auto inMax = get_input2<float>("Input max");
        auto outputMin = get_input2<float>("Output min");
        auto outputMax = get_input2<float>("Output max");
        auto clampMin = get_input2<bool>("Clamp min");
        auto clampMax = get_input2<bool>("Clamp max");
        auto curve = get_input<CurveObject>("Remap Ramp");
        auto ramp = get_input2<bool>("Use Ramp");
        if (scope == "vert"){
            if (prim->verts.attr_is<float>(attr)){
                auto &arr = prim->verts.attr<float>(attr);
                prim_remap<float>(arr, autoCompute, inMax, inMin, outputMax, outputMin, clampMax, clampMin, ramp, curve.get());
                }
            else if (prim->verts.attr_is<int>(attr)){
                auto &arr = prim->verts.attr<int>(attr);
                prim_remap<int>(arr, autoCompute, inMax, inMin, outputMax, outputMin, clampMax, clampMin, ramp, curve.get());
                }
            else{
                throw std::runtime_error("PrimAttrRemap: loops attr type not supported");
            }
        }
        else if (scope == "tri"){
            if (prim->tris.attr_is<float>(attr)){
                auto &arr = prim->tris.attr<float>(attr);
                prim_remap<float>(arr, autoCompute, inMax, inMin, outputMax, outputMin, clampMax, clampMin, ramp, curve.get());
                }
            else if (prim->tris.attr_is<int>(attr)){
                auto &arr = prim->tris.attr<int>(attr);
                prim_remap<int>(arr, autoCompute, inMax, inMin, outputMax, outputMin, clampMax, clampMin, ramp, curve.get());
                }
            else{
                throw std::runtime_error("PrimAttrRemap: loops attr type not supported");
            }
        }
        else if (scope == "loop"){
            if (prim->loops.attr_is<float>(attr)){
                auto &arr = prim->loops.attr<float>(attr);
                prim_remap<float>(arr, autoCompute, inMax, inMin, outputMax, outputMin, clampMax, clampMin, ramp, curve.get());
                }
            else if (prim->loops.attr_is<int>(attr)){
                auto &arr = prim->loops.attr<int>(attr);
                prim_remap<int>(arr, autoCompute, inMax, inMin, outputMax, outputMin, clampMax, clampMin, ramp, curve.get());
                }
            else{
                throw std::runtime_error("PrimAttrRemap: loops attr type not supported");
            }
        }
        else if (scope == "poly"){
            if (prim->polys.attr_is<float>(attr)){
                auto &arr = prim->polys.attr<float>(attr);
                prim_remap<float>(arr, autoCompute, inMax, inMin, outputMax, outputMin, clampMax, clampMin, ramp, curve.get());
                }
            else if (prim->polys.attr_is<int>(attr)){
                auto &arr = prim->polys.attr<int>(attr);
                prim_remap<int>(arr, autoCompute, inMax, inMin, outputMax, outputMin, clampMax, clampMin, ramp, curve.get());
                }
            else{
                throw std::runtime_error("PrimAttrRemap: loops attr type not supported");
            }
        }
        else if (scope == "line"){
            if (prim->lines.attr_is<float>(attr)){
                auto &arr = prim->lines.attr<float>(attr);
                prim_remap<float>(arr, autoCompute, inMax, inMin, outputMax, outputMin, clampMax, clampMin, ramp, curve.get());
                }
            else if (prim->lines.attr_is<int>(attr)){
                auto &arr = prim->lines.attr<int>(attr);
                prim_remap<int>(arr, autoCompute, inMax, inMin, outputMax, outputMin, clampMax, clampMin, ramp, curve.get());
                }
            else{
                throw std::runtime_error("PrimAttrRemap: loops attr type not supported");
            }
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimAttrRemap, {
    {
        {"PrimitiveObject", "prim", "", zeno::Socket_ReadOnly},
        {"enum vert tri loop poly line", "scope", "vert"},
        {"string", "attr", ""},
        {"bool", "Auto Compute input range", "0"},
        {"bool", "Clamp min", "0"},
        {"bool", "Clamp max", "0"},
        {"float", "Input min", "0"},
        {"float", "Input max", "1"},
        {"float", "Output min", "0"},
        {"float", "Output max", "1"},
        {"bool", "Use Ramp", "0"},
        {"curve", "Remap Ramp"},
        
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
