#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/HeatmapObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/scope_exit.h>
#include <stdexcept>
#include <cstring>
#include <zeno/utils/log.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include <tinygltf/stb_image.h>

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
#include "zeno/utils/string.h"

static const float eps = 0.0001f;

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
    int iu = int(uv[0] * (w-eps)) % w;
    if (iu < 0) {
        iu += w;
    }
    int iv = int(uv[1] * (h-eps)) % h;
    if (iv < 0) {
        iv += h;
    }
    return {iu, iv};
}
static zeno::vec2i uvClampToEdge(vec3f uv, int w, int h) {
    int iu = clamp(int(uv[0] * (w-eps)), 0, (w-1));
    int iv = clamp(int(uv[1] * (h-eps)), 0, (h-1));
    return {iu, iv};
}

//static zeno::vec3f queryColorInner(vec2i uv, const uint8_t* data, int w, int n) {
    //int iu = uv[0];
    //int iv = uv[1];
    //int start = (iu + iv * w) * n;
    //float r = float(data[start]) / 255.0f;
    //float g = float(data[start+1]) / 255.0f;
    //float b = float(data[start+2]) / 255.0f;
    //return {r, g, b};
//}
static zeno::vec3f queryColorInner(vec2i uv, const float* data, int w, int n) {
    int iu = uv[0];
    int iv = uv[1];
    int start = (iu + iv * w) * n;
    float r = (data[start]);
    float g = (data[start+1]);
    float b = (data[start+2]);
    return {r, g, b};
}
void primSampleTexture(
    std::shared_ptr<PrimitiveObject> prim,
    const std::string &srcChannel,
    const std::string &dstChannel,
    std::shared_ptr<PrimitiveObject> img,
    const std::string &wrap,
    // ZHOUHANG: please add arg filter, which is enum NEAREST LINEAR, impl bilerp version for LINEAR
    vec3f borderColor,
    float remapMin,
    float remapMax
) {
    if (!img->userData().has("isImage")) throw zeno::Exception("not an image");
    using ColorT = float;
    const ColorT *data = (float *)img->verts.data();
    auto &clr = prim->add_attr<zeno::vec3f>(dstChannel);
    auto &uv = prim->attr<zeno::vec3f>(srcChannel);
    auto w = img->userData().get2<int>("w");
    auto h = img->userData().get2<int>("h");
    std::function<zeno::vec3f(vec3f, const ColorT*, int, int, int, vec3f)> queryColor;
    // if (filter == "NEAREST") {
    if (wrap == "REPEAT") {
        queryColor = [=] (vec3f uv, const ColorT* data, int w, int h, int n, vec3f _clr)-> vec3f {
            uv = (uv - remapMin) / (remapMax - remapMin);
            auto iuv = uvRepeat(uv, w, h);
            return queryColorInner(iuv, data, w, n);
        };
    }
    else if (wrap == "CLAMP_TO_EDGE") {
        queryColor = [=] (vec3f uv, const ColorT* data, int w, int h, int n, vec3f _clr)-> vec3f {
            uv = (uv - remapMin) / (remapMax - remapMin);
            auto iuv = uvClampToEdge(uv, w, h);
            return queryColorInner(iuv, data, w, n);
        };
    }
    else if (wrap == "CLAMP_TO_BORDER") {
        queryColor = [=] (vec3f uv, const ColorT* data, int w, int h, int n, vec3f clr)-> vec3f {
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
    // else if (filter == "NEAREST") {
    // copy-paste all above wrap ifelses
    // }

    #pragma omp parallel for
    for (auto i = 0; i < uv.size(); i++) {
        clr[i] = queryColor(uv[i], data, w, h, 3, borderColor);
    }
}

struct PrimSample2D : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto srcChannel = get_input2<std::string>("uvChannel");
        auto dstChannel = get_input2<std::string>("targetChannel");
        auto image = get_input2<PrimitiveObject>("image");
        auto wrap = get_input2<std::string>("wrap");
        auto borderColor = get_input2<vec3f>("borderColor");
        auto remapMin = get_input2<float>("remapMin");
        auto remapMax = get_input2<float>("remapMax");
        primSampleTexture(prim, srcChannel, dstChannel, image, wrap, borderColor, remapMin, remapMax);

        set_output("outPrim", std::move(prim));
    }
};
ZENDEFNODE(PrimSample2D, {
    {
        {"PrimitiveObject", "prim"},
        {"PrimitiveObject", "image"},
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
std::shared_ptr<PrimitiveObject> readImageFile(std::string const &path) {
    int w, h, n;
    stbi_set_flip_vertically_on_load(true);
    float* data = stbi_loadf(path.c_str(), &w, &h, &n, 0);
    if (!data) {
        throw zeno::Exception("cannot open image file at path: " + path);
    }
    scope_exit delData = [=] { stbi_image_free(data); };
    auto img = std::make_shared<PrimitiveObject>();
    img->verts.resize(w * h * n);
    if (n == 3) {
        std::memcpy(img->verts.data(), data, w * h * n * sizeof(float));
    } else if (n == 4) {
        auto &alpha = img->verts.add_attr<float>("alpha");
        for (int i = 0; i < w * h; i++) {
            img->verts[i] = {data[i*4+0], data[i*4+1], data[i*4+2]};
            alpha[i] = data[i*4+3];
        }
    } else if (n == 2) {
        for (int i = 0; i < w * h; i++) {
            img->verts[i] = {data[i*2+0], data[i*2+1], 0};
        }
    } else if (n == 1) {
        for (int i = 0; i < w * h; i++) {
            img->verts[i] = vec3f(data[i*2+0]);
        }
    } else {
        throw zeno::Exception("too much number of channels");
    }
    img->userData().set2("isImage", 1);
    img->userData().set2("w", w);
    img->userData().set2("h", h);
    return img;
}

std::shared_ptr<PrimitiveObject> readExrFile(std::string const &path) {
    int nx, ny, nc = 4;
    float* rgba;
    const char* err;
    int ret = LoadEXR(&rgba, &nx, &ny, path.c_str(), &err);
    if (ret != 0) {
        zeno::log_error("load exr: {}", err);
        throw std::runtime_error(zeno::format("load exr: {}", err));
    }
    nx = std::max(nx, 1);
    ny = std::max(ny, 1);
//    for (auto i = 0; i < ny / 2; i++) {
//        for (auto x = 0; x < nx * 4; x++) {
//            auto index1 = i * (nx * 4) + x;
//            auto index2 = (ny - 1 - i) * (nx * 4) + x;
//            std::swap(rgba[index1], rgba[index2]);
//        }
//    }

    auto img = std::make_shared<PrimitiveObject>();
    img->verts.resize(nx * ny);

    auto &alpha = img->verts.add_attr<float>("alpha");
    for (int i = 0; i < nx * ny; i++) {
        img->verts[i] = {rgba[i*4+0], rgba[i*4+1], rgba[i*4+2]};
        alpha[i] = rgba[i*4+3];
    }
//
    img->userData().set2("isImage", 1);
    img->userData().set2("w", nx);
    img->userData().set2("h", ny);
    return img;
}

struct ReadImageFile : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        if (zeno::ends_with(path, ".exr", false)) {
            set_output("image", readExrFile(path));
        }
        else {
            set_output("image", readImageFile(path));
        }
    }
};
ZENDEFNODE(ReadImageFile, {
    {
        {"readpath", "path"},
    },
    {
        {"PrimitiveObject", "image"},
    },
    {},
    {"primitive"},
});
}
