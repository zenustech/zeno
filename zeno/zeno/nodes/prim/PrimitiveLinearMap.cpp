#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/logger.h>
#include <sstream>
#include <iostream>
#include <cmath>

namespace zeno {
    static int axisIndex(std::string const &axis) {
        if (axis.empty()) return 0;
        static const char *table[3] = {"X", "Y", "Z"};
        auto it = std::find(std::begin(table), std::end(table), axis);
        if (it == std::end(table)) throw std::runtime_error("invalid axis index: " + axis);
        return it - std::begin(table);
    }

    struct PrimitiveLinearMap : zeno::INode {
        virtual void apply() override {
            auto prim = get_input<zeno::PrimitiveObject>("prim");
            auto refPrim = get_input<zeno::PrimitiveObject>("refPrim");
            auto attrName = get_input<zeno::StringObject>("attrName")->get();
            auto refAttrNameSrc = get_input<zeno::StringObject>("refAttrNameSrc")->get();
            auto refAttrNameDst = get_input<zeno::StringObject>("refAttrNameDst")->get();
            auto axis = axisIndex(get_input<zeno::StringObject>("axis")->get());
            auto refAxisSrc = axisIndex(get_input<zeno::StringObject>("refAxisSrc")->get());
            auto refAxisDst = axisIndex(get_input<zeno::StringObject>("refAxisDst")->get());
            auto minVal = get_input<zeno::NumericObject>("minVal")->get<float>();
            auto maxVal = get_input<zeno::NumericObject>("maxVal")->get<float>();


            auto getAxis = [] (auto &val, int axis) -> auto & {
                using T = std::decay_t<decltype(val)>;
                if constexpr (std::is_arithmetic_v<T>) {
                    return val;
                } else {
                    return val[axis];
                }
            };

            refPrim->attr_visit(refAttrNameSrc, [&] (auto &refAttrSrc) {
                if (refAttrSrc.empty()) {
                    log_warn("src array is empty");
                    return;
                }

                std::vector<float> srcArr;
                srcArr.reserve(refAttrSrc.size() + 1);
                for (size_t i = 0; i < refAttrSrc.size(); i++) {
                    srcArr.push_back(getAxis(refAttrSrc[i], refAxisSrc));
                }
                srcArr.push_back(getAxis(refAttrSrc.back(), refAxisSrc));
                srcArr.push_back(getAxis(refAttrSrc.back(), refAxisSrc));

                refPrim->attr_visit(refAttrNameDst, [&] (auto &refAttrDst) {
                    if (refAttrDst.size() != refAttrSrc.size()) {
                        log_warn("dst and src size not equal");
                        return;
                    }

                    std::vector<float> dstArr;
                    dstArr.reserve(refAttrDst.size());
                    for (size_t i = 0; i < refAttrDst.size(); i++) {
                        dstArr.push_back(getAxis(refAttrDst[i], refAxisDst));
                    }
                    dstArr.push_back(getAxis(refAttrDst.back(), refAxisDst));
                    dstArr.push_back(getAxis(refAttrDst.back(), refAxisDst));

                    auto linmap = [&] (float src) {
                        auto it = std::lower_bound(srcArr.begin(), srcArr.end() - 2, src);
                        size_t index = it - srcArr.begin();
                        auto nit = it + 1;
                        auto fac = std::clamp((src - *it) / std::max(1e-8f, *nit - *it), 0.f, 1.f);
                        auto dst = dstArr[index] + (dstArr[index + 1] - dstArr[index]) * fac;
                        return dst;
                    };

                    prim->attr_visit(attrName, [&] (auto &attr) {
                        using T = std::decay_t<decltype(attr[0])>;

                        if (get_param<bool>("autoMinMax")) {
                            auto minv = getAxis(attr[0], axis);
                            auto maxv = getAxis(attr[0], axis);
#pragma omp parallel for reduction(min:minv) reduction(max:maxv)
                            for (intptr_t i = 0; i < attr.size(); ++i) {
                                auto val = getAxis(attr[i], axis);
                                maxv = std::max(maxv, val);
                                minv = std::min(minv, val);
                            }
                            minVal = minv + (maxv - minv) * minVal;
                            maxVal = minv + (maxv - minv) * maxVal;
                        }

#pragma omp parallel for
                        for (intptr_t i = 0; i < attr.size(); ++i) {
                            auto &val = getAxis(attr[i], axis);
                            val = linmap((val - minVal) / (maxVal - minVal));
                        }
                    });
                });
            });

            set_output("prim", std::move(prim));
        }
    };
ZENDEFNODE(PrimitiveLinearMap, {
    {
    {"PrimitiveObject", "prim"},
    {"PrimitiveObject", "refPrim"},
    {"string", "attrName", "pos"},
    {"string", "refAttrNameSrc", "pos"},
    {"string", "refAttrNameDst", "pos"},
    {"int", "minVal", "0"},
    {"int", "maxVal", "1"},
    {"enum X Y Z", "axis", "X"},
    {"enum X Y Z", "refAxisSrc", "X"},
    {"enum X Y Z", "refAxisDst", "Y"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    {"bool", "autoMinMax", "1"},
    },
    {"primitive"},
});
}
