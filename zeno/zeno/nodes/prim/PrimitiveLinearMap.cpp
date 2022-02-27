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
            auto attrNameDst = get_input<zeno::StringObject>("attrNameSrc")->get();
            auto attrNameSrc = get_input<zeno::StringObject>("attrNameDst")->get();
            auto refAttrNameSrc = get_input<zeno::StringObject>("refAttrNameSrc")->get();
            auto refAttrNameDst = get_input<zeno::StringObject>("refAttrNameDst")->get();
            auto axisSrc = axisIndex(get_input<zeno::StringObject>("axisSrc")->get());
            auto axisDst = axisIndex(get_input<zeno::StringObject>("axisDst")->get());
            auto refAxisSrc = axisIndex(get_input<zeno::StringObject>("refAxisSrc")->get());
            auto refAxisDst = axisIndex(get_input<zeno::StringObject>("refAxisDst")->get());
            auto limitMin = get_input<zeno::NumericObject>("limitMin")->get<float>();
            auto limitMax = get_input<zeno::NumericObject>("limitMax")->get<float>();
            auto autoMinMax = get_param<bool>("autoMinMax");
            auto autoSort = get_param<bool>("autoSort");

            auto getAxis = [] (auto &val, int axis) -> auto & {
                using T = std::decay_t<decltype(val)>;
                if constexpr (std::is_arithmetic_v<T>) {
                    return val;
                } else {
                    return val[axis];
                }
            };

            std::vector<float> srcArr;
            std::vector<float> dstArr;

            refPrim->attr_visit(refAttrNameSrc, [&] (auto &refAttrSrc) {
                if (refAttrSrc.empty()) {
                    log_warn("src array is empty");
                    return;
                }

                srcArr.reserve(refAttrSrc.size() + 2);
                srcArr.push_back(getAxis(refAttrSrc[0], refAxisSrc));
                for (size_t i = 0; i < refAttrSrc.size(); i++) {
                    srcArr.push_back(getAxis(refAttrSrc[i], refAxisSrc));
                }
                srcArr.push_back(getAxis(refAttrSrc.back(), refAxisSrc));

                std::vector<size_t> indices;
                indices.reserve(srcArr.size());
                for (size_t i = 0; i < srcArr.size(); i++) {
                    indices.push_back(i);
                }
                std::sort(indices.begin(), indices.end(), [&] (int pos1, int pos2) {
                    return srcArr[pos1] < srcArr[pos2];
                });

                if (autoSort) {
                    std::vector<float> srcArr2;
                    srcArr2.reserve(srcArr.size());
                    for (size_t i = 0; i < srcArr.size(); i++) {
                        srcArr2.push_back(srcArr[indices[i]]);
                    }
                    srcArr = std::move(srcArr2);
                }

                refPrim->attr_visit(refAttrNameDst, [&] (auto &refAttrDst) {
                    if (refAttrDst.size() != refAttrSrc.size()) {
                        log_warn("dst and src size not equal");
                        return;
                    }

                    dstArr.reserve(refAttrDst.size() + 2);
                    dstArr.push_back(getAxis(refAttrDst[0], refAxisDst));
                    for (size_t i = 0; i < refAttrDst.size(); i++) {
                        dstArr.push_back(getAxis(refAttrDst[i], refAxisDst));
                    }
                    dstArr.push_back(getAxis(refAttrDst.back(), refAxisDst));

                    if (autoSort) {
                        std::vector<float> dstArr2;
                        dstArr2.reserve(dstArr.size());
                        for (size_t i = 0; i < dstArr.size(); i++) {
                            dstArr2.push_back(dstArr[indices[i]]);
                        }
                        dstArr = std::move(dstArr2);
                    }
                });
            });

            auto linmap = [&] (float src) -> float {
                auto nit = std::lower_bound(srcArr.begin() + 1, srcArr.end() - 1, src);
                auto it = nit - 1;
                size_t index = it - srcArr.begin();
                float fac = (src - *it) / std::max(1e-8f, *nit - *it);
                float dst = dstArr[index] + (dstArr[index + 1] - dstArr[index]) * fac;
                return dst;
            };

            prim->attr_visit(attrNameSrc, [&] (auto &attrSrc) {
                if (autoMinMax) {
                    auto minv = getAxis(attrSrc[0], axisSrc);
                    auto maxv = getAxis(attrSrc[0], axisSrc);
#ifndef _WIN32
#pragma omp parallel for reduction(min:minv) reduction(max:maxv)
#endif
                    for (intptr_t i = 0; i < attrSrc.size(); ++i) {
                        auto val = getAxis(attrSrc[i], axisSrc);
                        maxv = std::max(maxv, val);
                        minv = std::min(minv, val);
                    }
                    limitMin = minv + (maxv - minv) * limitMin;
                    limitMax = minv + (maxv - minv) * limitMax;
                }

                prim->attr_visit(attrNameDst, [&] (auto &attrDst) {
#pragma omp parallel for
                    for (intptr_t i = 0; i < attrSrc.size(); ++i) {
                        auto src = getAxis(attrSrc[i], axisSrc);
                        auto dst = linmap((src - limitMin) / (limitMax - limitMin));
                        dst = dst * (limitMax - limitMin) + limitMin;
                        getAxis(attrDst[i], axisDst) = dst;
                    }
                });
            });

            set_output("prim", std::move(prim));
        }
    };
ZENDEFNODE(PrimitiveLinearMap, {
    {
    {"PrimitiveObject", "prim"},
    {"PrimitiveObject", "refPrim"},
    {"string", "attrNameSrc", "pos"},
    {"string", "attrNameDst", "pos"},
    {"string", "refAttrNameSrc", "pos"},
    {"string", "refAttrNameDst", "pos"},
    {"float", "limitMin", "0"},
    {"float", "limitMax", "1"},
    {"enum X Y Z", "axisSrc", "X"},
    {"enum X Y Z", "axisDst", "Y"},
    {"enum X Y Z", "refAxisSrc", "X"},
    {"enum X Y Z", "refAxisDst", "Y"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    {"bool", "autoMinMax", "1"},
    {"bool", "autoSort", "1"},
    },
    {"primitive"},
});
}
