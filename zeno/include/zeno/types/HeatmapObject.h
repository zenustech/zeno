#pragma once
//
// Created by zhouhang on 2022/12/14.
//

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
