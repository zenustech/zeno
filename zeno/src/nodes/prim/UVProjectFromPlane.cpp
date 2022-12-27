#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <stdexcept>
#include "zeno/utils/log.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include <tinygltf/stb_image.h>

namespace zeno {
struct UVProjectFromPlane : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto &uv = prim->verts.add_attr<vec3f>("uv");
        auto refPlane = get_input<PrimitiveObject>("refPlane");
        if (refPlane->verts.size() != 4) {
            zeno::log_error("refPlane must be 1 * 1 plane!");
            throw zeno::makeError("refPlane must be 1 * 1 plane!");
        }
        auto originPos = refPlane->verts[2];
        auto xOffset = refPlane->verts[0];
        auto yOffset = refPlane->verts[3];
//        zeno::log_info("xOffset:{}, originPos: {}", xOffset, originPos);
        auto uDir = zeno::normalize(xOffset - originPos);
        auto vDir = zeno::normalize(yOffset - originPos);
        auto uLength = zeno::length(xOffset - originPos);
        auto vLength = zeno::length(yOffset - originPos);
//        zeno::log_info("uDir:{], uLength: {}, n: {}", uDir, uLength);
        for (auto i = 0; i < prim->size(); i++) {
            auto &vert = prim->verts[i];
            auto offset = vert - originPos;
            auto proj = offset;
            auto u = zeno::clamp(zeno::dot(proj, uDir) / uLength, 0, 1);
            auto v = zeno::clamp(zeno::dot(proj, vDir) / vLength, 0, 1);
            uv[i] = zeno::vec3f(u,  v, 0);
        }
        auto &uv0 = prim->tris.add_attr<vec3f>("uv0");
        auto &uv1 = prim->tris.add_attr<vec3f>("uv1");
        auto &uv2 = prim->tris.add_attr<vec3f>("uv2");
        for (auto i = 0; i < prim->tris.size(); i++) {
            auto tri = prim->tris[i];
            uv0[i] = uv[tri[0]];
            uv1[i] = uv[tri[1]];
            uv2[i] = uv[tri[2]];
        }

        if(prim->loops.size()){
            prim->loops.add_attr<int>("uvs");
            for (auto i = 0; i < prim->loops.size(); i++) {
                auto lo = prim->loops[i];
                prim->loops.attr<int>("uvs")[i] = lo;
            }
            prim->uvs.resize(prim->size());
            for (auto i = 0; i < prim->size(); i++) {
                prim->uvs[i] = {uv[i][0], uv[i][1]};
            }
        }
        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(UVProjectFromPlane, {
    {
        {"PrimitiveObject", "prim"},
        {"PrimitiveObject", "refPlane"},
    },
    {
        {"PrimitiveObject", "outPrim"}
    },
    {},
    {"primitive"},
});

static zeno::vec2i uvRepeat(vec3f uv, int w, int h) {
    int iu = int(uv[0] * (w-1)) % w;
    if (iu < 0) {
        iu += w;
    }
    int iv = int(uv[1] * (h-1)) % h;
    if (iv < 0) {
        iv += h;
    }
    return {iu, iv};
}
static zeno::vec2i uvClampToEdge(vec3f uv, int w, int h) {
    int iu = clamp(int(uv[0] * (w-1)), 0, (w-1));
    int iv = clamp(int(uv[1] * (h-1)), 0, (h-1));
    return {iu, iv};
}

static zeno::vec3f queryColorInner(vec2i uv, const uint8_t* data, int w, int n) {
    int iu = uv[0];
    int iv = uv[1];
    int start = (iu + iv * w) * n;
    float r = float(data[start]) / 255.0f;
    float g = float(data[start+1]) / 255.0f;
    float b = float(data[start+2]) / 255.0f;
    return {r, g, b};
}
void primSampleTexture(
    std::shared_ptr<PrimitiveObject> prim,
    const std::string &srcChannel,
    const std::string &dstChannel,
    const std::string &imagePath,
    const std::string &wrap,
    vec3f borderColor,
    float remapMin,
    float remapMax
) {
    int w, h, n;
    stbi_set_flip_vertically_on_load(true);
    uint8_t* data = stbi_load(imagePath.c_str(), &w, &h, &n, 0);
    auto &clr = prim->add_attr<zeno::vec3f>(dstChannel);
    auto &uv = prim->attr<zeno::vec3f>(srcChannel);
    std::function<zeno::vec3f(vec3f, const uint8_t*, int, int, int, vec3f)> queryColor;
    if (wrap == "REPEAT") {
        queryColor = [=] (vec3f uv, const uint8_t* data, int w, int h, int n, vec3f _clr)-> vec3f {
            uv = (uv - remapMin) / (remapMax - remapMin);
            auto iuv = uvRepeat(uv, w, h);
            return queryColorInner(iuv, data, w, n);
        };
    }
    else if (wrap == "CLAMP_TO_EDGE") {
        queryColor = [=] (vec3f uv, const uint8_t* data, int w, int h, int n, vec3f _clr)-> vec3f {
            uv = (uv - remapMin) / (remapMax - remapMin);
            auto iuv = uvClampToEdge(uv, w, h);
            return queryColorInner(iuv, data, w, n);
        };
    }
    else if (wrap == "CLAMP_TO_BORDER") {
        queryColor = [=] (vec3f uv, const uint8_t* data, int w, int h, int n, vec3f clr)-> vec3f {
            uv = (uv - remapMin) / (remapMax - remapMin);
            if (uv[0] < 0 || uv[0] > 1 || uv[1] < 0 || uv[1] > 1) {
                return clr;
            }
            auto iuv = uvClampToEdge(uv, w, h);
            return queryColorInner(iuv, data, w, n);
        };
    }
    else {
        zeno::log_error("wrap type error");
        throw std::runtime_error("wrap type error");
    }

    #pragma omp parallel for
    for (auto i = 0; i < uv.size(); i++) {
        clr[i] = queryColor(uv[i], data, w, h, n, borderColor);
    }
    stbi_image_free(data);
}

struct PrimSample2D : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto srcChannel = get_input2<std::string>("uvChannel");
        auto dstChannel = get_input2<std::string>("targetChannel");
        auto imagePath = get_input2<std::string>("imagePath");
        auto wrap = get_input2<std::string>("wrap");
        auto borderColor = get_input2<vec3f>("borderColor");
        auto remapMin = get_input2<float>("remapMin");
        auto remapMax = get_input2<float>("remapMax");
        primSampleTexture(prim, srcChannel, dstChannel, imagePath, wrap, borderColor, remapMin, remapMax);

        set_output("outPrim", std::move(prim));
    }
};
ZENDEFNODE(PrimSample2D, {
    {
        {"PrimitiveObject", "prim"},
        {"readpath", "imagePath"},
        {"string", "uvChannel", "uv"},
        {"string", "targetChannel", "clr"},
        {"float", "remapMin", "0"},
        {"float", "remapMax", "1"},
        {"enum REPEAT CLAMP_TO_EDGE CLAMP_TO_BORDER", "wrap", "REPEAT"},
        {"vec3f", "borderColor", "0,0,0"},
    },
    {
        {"PrimitiveObject", "outPrim"}
    },
    {},
    {"primitive"},
});
}
