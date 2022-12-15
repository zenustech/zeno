//
// Created by zhouhang on 2022/12/14.
//

#ifndef ZENO_HEATMAPOBJECT_H
#define ZENO_HEATMAPOBJECT_H
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>

namespace zeno {
    struct HeatmapObject : zeno::IObject {
        std::vector<zeno::vec3f> colors;
        zeno::vec3f interp(float x) const {
            x = zeno::clamp(x, 0, 1) * colors.size();
            int i = (int) zeno::floor(x);
            i = zeno::clamp(i, 0, colors.size() - 2);
            float f = x - i;
            return (1 - f) * colors.at(i) + f * colors.at(i + 1);
        }
    };
    inline void primSampleHeatmap(
            std::shared_ptr<PrimitiveObject> prim,
            const std::string &srcChannel,
            const std::string &dstChannel,
            std::shared_ptr<HeatmapObject> heatmap,
            float remapMin,
            float remapMax
    ) {
        auto &clr = prim->add_attr<zeno::vec3f>(dstChannel);
        auto &src = prim->attr<float>(srcChannel);
#pragma omp parallel for //ideally this could be done in opengl
        for (int i = 0; i < src.size(); i++) {
            auto x = (src[i]-remapMin)/(remapMax-remapMin);
            clr[i] = heatmap->interp(x);
        }
    }
    inline void primSampleTexture(
        std::shared_ptr<PrimitiveObject> prim,
        const std::string &srcChannel,
        const std::string &dstChannel,
        const std::string &imagePath,
        const std::string &wrap,
        vec3f borderColor,
        float remapMin,
        float remapMax
    );
}
#endif //ZENO_HEATMAPOBJECT_H
