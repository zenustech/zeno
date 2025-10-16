#pragma once
//
// Created by zhouhang on 2022/12/14.
//

#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <reflect/core.hpp>

namespace zeno {
    struct HeatmapObject : IObjectClone<HeatmapObject> {

        void Delete() override {
            //delete this;
        }

        std::vector<zeno::vec3f> colors;
        zeno::vec3f interp(float x) const {
            if(x <= 0) return colors[0];
            if(x >= 1) return colors[colors.size() - 1];
            x = zeno::clamp(x, 0, 1) * (colors.size()-1);
            int i = (int) zeno::floor(x);
            float f = x - i;
            return zeno::mix(colors[i], colors[i + 1], f);
        }
    };
    void primSampleHeatmap(
            PrimitiveObject* prim,
            const std::string &srcChannel,
            const std::string &dstChannel,
            HeatmapObject* heatmap,
            float remapMin,
            float remapMax
    );



    struct HeatmapData2 {
        //TODO
        std::vector<zeno::vec3f> colors;
    };
}
