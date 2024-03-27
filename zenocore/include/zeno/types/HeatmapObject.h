#pragma once
//
// Created by zhouhang on 2022/12/14.
//

#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>

namespace zeno {
    struct HeatmapObject : IObjectClone<HeatmapObject> {
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
            std::shared_ptr<PrimitiveObject> prim,
            const std::string &srcChannel,
            const std::string &dstChannel,
            std::shared_ptr<HeatmapObject> heatmap,
            float remapMin,
            float remapMax
    );
    std::shared_ptr<PrimitiveObject> readExrFile(std::string const &path);
    ZENO_API std::shared_ptr<PrimitiveObject> readImageFile(std::string const &path);
    ZENO_API std::shared_ptr<PrimitiveObject> readPFMFile(std::string const &path);
    ZENO_API void write_pfm(std::string& path, std::shared_ptr<PrimitiveObject> image);
    ZENO_API void write_jpg(std::string& path, std::shared_ptr<PrimitiveObject> image);
    void primSampleTexture(
        std::shared_ptr<PrimitiveObject> prim,
        const std::string &srcChannel,
        const std::string &srcSource,
        const std::string &dstChannel,
        std::shared_ptr<PrimitiveObject> img,
        const std::string &wrap,
        vec3f borderColor,
        float remapMin,
        float remapMax
    );
    void primSampleTexture(
        std::shared_ptr<PrimitiveObject> prim,
        const std::string &srcChannel,
        const std::string &uvSource,
        const std::string &dstChannel,
        std::shared_ptr<PrimitiveObject> img,
        const std::string &wrap,
        const std::string &filter,
        vec3f borderColor,
        float remapMin,
        float remapMax
    );
}
