#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/HeatmapObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/scope_exit.h>
#include <stdexcept>
#include <filesystem>
#include <cstring>
#include <zeno/utils/log.h>
#include <zeno/utils/fileio.h>
#include <zeno/utils/image_proc.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include <tinygltf/stb_image.h>
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
#include "zeno/utils/string.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include <tinygltf/stb_image_write.h>
#include <vector>
#include <zeno/types/HeatmapObject.h>

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
void primSampleTexture(
        std::shared_ptr<PrimitiveObject> prim,
        const std::string &srcChannel,
        const std::string &uvSource,
        const std::string &dstChannel,
        std::shared_ptr<PrimitiveObject> img,
        const std::string &wrap,
        vec3f borderColor,
        float remapMin,
        float remapMax
) {
}

static vec2i uv_to_tex(vec2f texCoord, int w, int h) {
    int x = (int)(texCoord[0] * w - 0.5f) % w;
    int y = (int)(texCoord[1] * h - 0.5f) % h;
    x = x < 0 ? w + x : x;
    y = y < 0 ? h + y : y;
    return {x, y};
}

static vec3f getColor(vec2i tex, const vec3f* data, int w, int h) {
    tex[0] = tex[0] % w;
    tex[1] = tex[1] % h;
    int index = tex[1] * w + tex[0];
    return data[index];
}

static vec3f getColorClamp(vec2i tex, const vec3f* data, int w, int h) {
    tex[0] = zeno::clamp(tex[0], 0, w-1);
    tex[1] = zeno::clamp(tex[1], 0, h-1);
    int index = tex[1] * w + tex[0];
    return data[index];
}

static vec3f Sample2DLinear(vec2f texCoord, const vec3f* data, int w, int h) {
    texCoord = texCoord * vec2f(w, h) - vec2f(0.5f);
    vec2f f = fract(texCoord);
    int x = (int)(texCoord[0]) % w;
    int y = (int)(texCoord[1]) % h;
    x = x < 0 ? w + x : x;
    y = y < 0 ? h + y : y;
    vec3f s1 = getColor(vec2i(x,y), data, w, h);
    vec3f s2 = getColor(vec2i(x+1,y), data, w, h);
    vec3f s3 = getColor(vec2i(x,y+1), data, w, h);
    vec3f s4 = getColor(vec2i(x+1,y+1), data, w, h);
    return mix(mix(s1, s2, f[0]), mix(s3, s4, f[0]), f[1]);
}

static vec3f Sample2DLinearClamp(vec2f texCoord, const vec3f* data, int w, int h) {
    texCoord = texCoord * vec2f(w, h) - vec2f(0.5f);
    vec2f f = fract(texCoord);
    int x = (int)(texCoord[0]) % w;
    int y = (int)(texCoord[1]) % h;
    x = x < 0 ? w + x : x;
    y = y < 0 ? h + y : y;
    vec3f s1 = getColorClamp(vec2i(x,y), data, w, h);
    vec3f s2 = getColorClamp(vec2i(x+1,y), data, w, h);
    vec3f s3 = getColorClamp(vec2i(x,y+1), data, w, h);
    vec3f s4 = getColorClamp(vec2i(x+1,y+1), data, w, h);
    return mix(mix(s1, s2, f[0]), mix(s3, s4, f[0]), f[1]);
}
struct PrimSample2D : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto srcChannel = get_input2<std::string>("uvChannel");
        auto uvSource = get_input2<std::string>("uvSource");
        auto dstChannel = get_input2<std::string>("targetChannel");
        auto image = get_input2<PrimitiveObject>("image");
        auto wrap = get_input2<std::string>("wrap");
        auto filter = get_input2<std::string>("filter");
        auto borderColor = get_input2<vec3f>("borderColor");

        if (!image->userData().has("isImage")) {
            throw zeno::Exception("not an image");
        }
        auto w = image->userData().get2<int>("w");
        auto h = image->userData().get2<int>("h");

        auto &clrs = prim->add_attr<zeno::vec3f>(dstChannel);
        auto data = image->verts.data();
        std::function<zeno::vec3f(vec3f, const vec3f*, int, int, vec3f)> queryColor;
        if (filter == "nearest") {
            if (wrap == "REPEAT") {
                queryColor = [=](vec3f uv3, const vec3f *data, int w, int h, vec3f clr) -> vec3f {
                    vec2f uv = {uv3[0], uv3[1]};
                    vec2i texuv = uv_to_tex(uv, w, h);
                    return getColor(texuv, data, w, h);
                };
            }
            else if (wrap == "CLAMP_TO_EDGE") {
                queryColor = [=](vec3f uv3, const vec3f *data, int w, int h, vec3f clr) -> vec3f {
                    vec2f uv = {uv3[0], uv3[1]};
                    uv = zeno::clamp(uv, 0, 1);
                    vec2i texuv = uv_to_tex(uv, w, h);
                    return getColor(texuv, data, w, h);
                };
            }
            else if (wrap == "CLAMP_TO_BORDER") {
                queryColor = [=](vec3f uv3, const vec3f *data, int w, int h, vec3f clr) -> vec3f {
                    vec2f uv = {uv3[0], uv3[1]};
                    uv = zeno::clamp(uv, 0, 1);
                    vec2i texuv = uv_to_tex(uv, w, h);
                    if (uv[0] != uv3[0] || uv[1] != uv3[1]) {
                        return borderColor;
                    } else {
                        return getColor(texuv, data, w, h);
                    }
                };
            }
        }
        else {
            if (wrap == "REPEAT") {
                queryColor = [=](vec3f uv3, const vec3f *data, int w, int h, vec3f clr) -> vec3f {
                    vec2f uv = {uv3[0], uv3[1]};
                    return Sample2DLinear(uv, data, w, h);
                };
            }
            else if (wrap == "CLAMP_TO_EDGE") {
                queryColor = [=](vec3f uv3, const vec3f *data, int w, int h, vec3f clr) -> vec3f {
                    vec2f uv = {uv3[0], uv3[1]};
                    uv = zeno::clamp(uv, 0, 1);
                    return Sample2DLinearClamp(uv, data, w, h);
                };
            }
            else if (wrap == "CLAMP_TO_BORDER") {
                queryColor = [=](vec3f uv3, const vec3f *data, int w, int h, vec3f clr) -> vec3f {
                    vec2f uv = {uv3[0], uv3[1]};
                    uv = zeno::clamp(uv, 0, 1);
                    if ((uv[0] != uv3[0] || uv[1] != uv3[1])) {
                        return borderColor;
                    }
                    else {
                        return Sample2DLinearClamp(uv, data, w, h);
                    }
                };
            }
        }

        if (uvSource == "vertex") {
            auto &uv = prim->attr<zeno::vec3f>(srcChannel);
            #pragma omp parallel for
            for (auto i = 0; i < uv.size(); i++) {
                clrs[i] = queryColor(uv[i], data, w, h, borderColor);
            }
        }
        else if (uvSource == "tris") {
            auto uv0 = prim->tris.attr<vec3f>("uv0");
            auto uv1 = prim->tris.attr<vec3f>("uv1");
            auto uv2 = prim->tris.attr<vec3f>("uv2");
            #pragma omp parallel for
            for (auto i = 0; i < prim->tris.size(); i++) {
                auto tri = prim->tris[i];
                clrs[tri[0]] = queryColor(uv0[i], data, w, h, borderColor);
                clrs[tri[1]] = queryColor(uv1[i], data, w, h, borderColor);
                clrs[tri[2]] = queryColor(uv2[i], data, w, h, borderColor);
            }

        }
        else if (uvSource == "loopsuv") {
            auto &loopsuv = prim->loops.attr<int>("uvs");
            #pragma omp parallel for
            for (auto i = 0; i < prim->loops.size(); i++) {
                auto uv = prim->uvs[loopsuv[i]];
                int index = prim->loops[i];
                clrs[index] = queryColor({uv[0], uv[1], 0}, data, w, h, borderColor);
            }
        }
        else {
            zeno::log_error("unknown uvSource");
            throw std::runtime_error("unknown uvSource");
        }

        set_output("outPrim", std::move(prim));
    }
};
ZENDEFNODE(PrimSample2D, {
    {
        {"PrimitiveObject", "prim"},
        {"PrimitiveObject", "image"},
        {"string", "uvChannel", "uv"},
        {"enum vertex tris loopsuv", "uvSource", "vertex"},
        {"string", "targetChannel", "clr"},
        {"float", "remapMin", "0"},
        {"float", "remapMax", "1"},
        {"enum REPEAT CLAMP_TO_EDGE CLAMP_TO_BORDER", "wrap", "REPEAT"},
        {"enum nearest linear", "filter", "nearest"},
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
    std::string native_path = std::filesystem::u8path(path).string();
    float* data = stbi_loadf(native_path.c_str(), &w, &h, &n, 0);
    if (!data) {
        throw zeno::Exception("cannot open image file at path: " + native_path);
    }
    scope_exit delData = [=] { stbi_image_free(data); };
    auto img = std::make_shared<PrimitiveObject>();
    img->verts.resize(w * h);
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
            img->verts[i] = vec3f(data[i]);
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
    std::string native_path = std::filesystem::u8path(path).string();
    int ret = LoadEXR(&rgba, &nx, &ny, native_path.c_str(), &err);
    if (ret != 0) {
        zeno::log_error("load exr: {}", err);
        throw std::runtime_error(zeno::format("load exr: {}", err));
    }
    nx = std::max(nx, 1);
    ny = std::max(ny, 1);
    for (auto i = 0; i < ny / 2; i++) {
        for (auto x = 0; x < nx * 4; x++) {
            auto index1 = i * (nx * 4) + x;
            auto index2 = (ny - 1 - i) * (nx * 4) + x;
            std::swap(rgba[index1], rgba[index2]);
        }
    }

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

std::shared_ptr<PrimitiveObject> readPFMFile(std::string const &path) {
    int nx = 0;
    int ny = 0;
    std::ifstream file(path, std::ios::binary);
    std::string format;
    file >> format;
    file >> nx >> ny;
    float scale = 0;
    file >> scale;
    file.ignore(1);

    auto img = std::make_shared<PrimitiveObject>();
    int size = nx * ny;
    img->resize(size);
    file.read(reinterpret_cast<char*>(img->verts.data()), sizeof(vec3f) * nx * ny);

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
        else if (zeno::ends_with(path, ".pfm", false)) {
            set_output("image", readPFMFile(path));
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
    {"comp"},
});

template<typename T>
void image_flip_vertical(T *v, int w, int h) {
    for (auto j = 0; j < h / 2; j++) {
        for (auto i = 0; i < w; i++) {
            auto index1 = i + j * w;
            auto index2 = i + (h - j - 1) * w;
            std::swap(v[index1], v[index2]);
        }
    }
}

struct ImageFlipVertical : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        image_flip_vertical(image->verts.data(), w, h);
        if (image->verts.has_attr("alpha")) {
            auto alpha = image->verts.attr<float>("alpha");
            image_flip_vertical(alpha.data(), w, h);
        }
        set_output("image", image);
    }
};
ZENDEFNODE(ImageFlipVertical, {
    {
        {"image"},
    },
    {
        {"image"},
    },
    {},
    {"comp"},
});

void write_pfm(std::string& path, int w, int h, vec3f *rgb) {
    std::string header = zeno::format("PF\n{} {}\n-1.0\n", w, h);
    std::vector<char> data(header.size() + w * h * sizeof(vec3f));
    memcpy(data.data(), header.data(), header.size());
    memcpy(data.data() + header.size(), rgb, w * h * sizeof(vec3f));
    file_put_binary(data, path);
}

void write_pfm(std::string& path, std::shared_ptr<PrimitiveObject> image) {
    auto &ud = image->userData();
    int w = ud.get2<int>("w");
    int h = ud.get2<int>("h");
    write_pfm(path, w, h, image->verts->data());
}

void write_jpg(std::string& path, std::shared_ptr<PrimitiveObject> image) {
    int w = image->userData().get2<int>("w");
    int h = image->userData().get2<int>("h");
    std::vector<uint8_t> colors;
    for (auto i = 0; i < w * h; i++) {
        auto rgb = zeno::pow(image->verts[i], 1.0f / 2.2f);
        int r = zeno::clamp(int(rgb[0] * 255.99), 0, 255);
        int g = zeno::clamp(int(rgb[1] * 255.99), 0, 255);
        int b = zeno::clamp(int(rgb[2] * 255.99), 0, 255);
        colors.push_back(r);
        colors.push_back(g);
        colors.push_back(b);
    }
    stbi_flip_vertically_on_write(1);
    stbi_write_jpg(path.c_str(), w, h, 3, colors.data(), 100);
}

struct WriteImageFile : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto path = get_input2<std::string>("path");
        auto type = get_input2<std::string>("type");
        auto boolgamma = get_input2<bool>("gamma");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        int n = 4;
        auto A = std::make_shared<PrimitiveObject>();
        A->verts.resize(image->size());
        A->verts.add_attr<float>("alpha");
        for(int i = 0;i < w * h;i++){
            A->verts.attr<float>("alpha")[i] = 1.0;
        }
        std::vector<float> &alpha = A->verts.attr<float>("alpha");
        if(image->verts.has_attr("alpha")){
            n = 4;
            alpha = image->verts.attr<float>("alpha");
        }
        if(has_input("mask")) {
            n = 4;
            auto mask = get_input2<PrimitiveObject>("mask");
            image->verts.add_attr<float>("alpha");
            image->verts.attr<float>("alpha") = mask->verts.attr<float>("alpha");
            alpha = mask->verts.attr<float>("alpha");
        }
        std::vector<char> data(w * h * n);
        float gamma = 1;
        if(boolgamma){
            gamma = 2.2f;
        }
        for (int i = 0; i < w * h; i++) {
            data[n * i + 0] = (char)(255 * powf(image->verts[i][0], 1.0f / gamma));
            data[n * i + 1] = (char)(255 * powf(image->verts[i][1], 1.0f / gamma));
            data[n * i + 2] = (char)(255 * powf(image->verts[i][2], 1.0f / gamma));
            data[n * i + 3] = (char)(255 * alpha[i]);
        }
        if(type == "jpg"){
            path += ".jpg";
            std::string native_path = std::filesystem::u8path(path).string();
            stbi_flip_vertically_on_write(1);
            stbi_write_jpg(native_path.c_str(), w, h, n, data.data(), 100);
        }
        else if(type == "png"){
            path += ".png";
            std::string native_path = std::filesystem::u8path(path).string();
            stbi_flip_vertically_on_write(1);
            stbi_write_png(native_path.c_str(), w, h, n, data.data(),0);
        }
        else if(type == "exr"){
            std::vector<float> data2(w * h * n);
            constexpr float gamma = 2.2f;
            for (int i = 0; i < w * h; i++) {
                data2[n * i + 0] = image->verts[i][0];
                data2[n * i + 1] = image->verts[i][1];
                data2[n * i + 2] = image->verts[i][2];
                data2[n * i + 3] = alpha[i];
            }
            for (auto i = 0; i < h / 2; i++) {
                for (auto x = 0; x < w * 4; x++) {
                    auto index1 = i * (w * 4) + x;
                    auto index2 = (h - 1 - i) * (w * 4) + x;
                    std::swap(data2[index1], data2[index2]);
                }
            }

            // Create EXR header
            EXRHeader header;
            InitEXRHeader(&header);

            // Set image width, height, and number of channels
            header.num_channels = n;

            // Create EXR image
            EXRImage exrImage;
            InitEXRImage(&exrImage);

            // Set image data
            exrImage.num_channels = n;
            exrImage.width = w;
            exrImage.height = h;
            exrImage.images = reinterpret_cast<unsigned char**>(&data2[0]);

            // Set image channel names (optional)
            std::vector<std::string> channelNames = {"R", "G", "B", "A"};
            header.channels = new EXRChannelInfo[n];
            for (int i = 0; i < n; ++i) {
                strncpy(header.channels[i].name, channelNames[i].c_str(), 255);
                header.channels[i].name[strlen(channelNames[i].c_str())] = '\0';
                header.channels[i].pixel_type = TINYEXR_PIXELTYPE_FLOAT;
            }

            const char* err;
            path += ".exr";
            std::string native_path = std::filesystem::u8path(path).string();
            int ret = SaveEXR(data2.data(),w,h,n,0,native_path.c_str(),&err);

            if (ret != TINYEXR_SUCCESS) {
                zeno::log_error("Error saving EXR file: %s\n", err);
                FreeEXRErrorMessage(err); // free memory allocated by the library
                return;
            }
            else{
                zeno::log_info("EXR file saved successfully.");
            }
        }
        else if (type == "pfm") {
            path = path + ".pfm";
            write_pfm(path, image);
        }
        set_output("image", image);
    }
};
ZENDEFNODE(WriteImageFile, {
    {
        {"image"},
        {"writepath", "path"},
        {"enum png jpg exr pfm", "type", "png"},
        {"mask"},
        {"bool", "gamma", "1"},
    },
    {
        {"image"},
    },
    {},
    {"comp"},
});

std::vector<zeno::vec3f> float_gaussian_blur(const vec3f *data, int w, int h) {
    float weight[5] = {0.227027f, 0.1945946f, 0.1216216f, 0.054054f, 0.016216f};
    std::vector<zeno::vec3f> img_pass(w * h);

#pragma omp parallel for
    for (auto j = 0; j < h; j++) {
        for (auto i = 0; i < w; i++) {
            vec3f sum = {};
            int index = i + w * j;
            for (auto k = 0; k < 5; k++) {
                if (k == 0) {
                    sum += data[index] * weight[k];
                }
                else {
                    int index_r = (i + k + w) % w + w * j;
                    sum += data[index_r] * weight[k];
                    int index_l = (i - k + w) % w + w * j;
                    sum += data[index_l] * weight[k];
                }
            }
            img_pass[index] = sum;
        }
    }

    std::vector<zeno::vec3f> img_out(w * h);

#pragma omp parallel for
    for (auto j = 0; j < h; j++) {
        for (auto i = 0; i < w; i++) {
            vec3f sum = {};
            int index = i + w * j;
            for (auto k = 0; k < 5; k++) {
                if (k == 0) {
                    sum += img_pass[index] * weight[k];
                }
                else {
                    int index_t = i + w * ((j + k + h) % h);
                    sum += img_pass[index_t] * weight[k];
                    int index_b = i + w * ((j - k + h) % h);
                    sum += img_pass[index_b] * weight[k];
                }
            }
            img_out[index] = sum;
        }
    }
    return img_out;
}

struct ImageFloatGaussianBlur : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        auto img_out = std::make_shared<PrimitiveObject>();
        img_out->resize(w * h);
        img_out->userData().set2("w", w);
        img_out->userData().set2("h", h);
        img_out->userData().set2("isImage", 1);
        img_out->verts.values = float_gaussian_blur(image->verts.data(), w, h);

        set_output("image", img_out);
    }
};

ZENDEFNODE(ImageFloatGaussianBlur, {
    {
        {"image"},
    },
    {
        {"image"},
    },
    {},
    {"comp"},
});
}
