#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/IGeometryObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/utils/parallel_reduce.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <stdexcept>
#include <cmath>
#include <zeno/utils/log.h>
#include <filesystem>


namespace zeno {

namespace {

template <class T>
static T BlendMode(const float &alpha1, const float &alpha2, const T& rgb1, const T& rgb2, const T& background, const vec3f opacity, std::string compmode)//rgb1 and background is premultiplied?
{
        if(compmode == std::string("Copy")) {//copy and over is different!
                T value = rgb1 * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Over")) {
                T value = (rgb1 + background * (1 - alpha1)) * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Under")) {
                T value = (background + rgb1 * (1 - alpha2)) * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Atop")) {
                T value = (rgb1 * alpha2 + background * (1 - alpha1)) * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("In")) {
                T value = rgb1 * alpha2 * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Out")) {
                T value = (rgb1 * (1 - alpha2)) * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Xor")) {
                T value = (rgb1 * (1 - alpha2) + background * (1 - alpha1)) * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Add")) {
                T value = (rgb1 + background) * opacity[0] + rgb2 * (1 - opacity[0]);//clamp?
                return value;
        }
        else if(compmode == std::string("Subtract")) {
                T value = (background - rgb1) * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Multiply")) {
                T value = rgb1 * background * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Max(Lighten)")) {
                T value = zeno::max(rgb1, background) * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Min(Darken)")) {
                T value = zeno::min(rgb1, background) * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Screen")) {//A+B-AB if A and B between 0-1, else A if A>B else B
                    T value = (1 - (1 - background) * (1 - rgb1)) * opacity[0] + rgb2 * (1 - opacity[0]);//only care 0-1!
                    return value;
        }
        else if(compmode == std::string("Difference")) {
                    T value = zeno::abs(rgb1 - background) * opacity[0] + rgb2 * (1 - opacity[0]);
                    return value;
        }
        else if(compmode == std::string("Average")) {
                    T value = (rgb1 + background) / 2 * opacity[0] + rgb2 * (1 - opacity[0]);
                    return value;
        }
        return T(0);
}

static zeno::vec3f BlendModeV(const float &alpha1, const float &alpha2, const vec3f& rgb1, const vec3f& rgb2, const vec3f& background, const vec3f opacity, std::string compmode)
{
        if(compmode == std::string("Overlay")) {
                    vec3f value;
                    for (int k = 0; k < 3; k++) {
                        if (rgb2[k] < 0.5) {
                            value[k] = 2 * rgb1[k] * background[k];
                        } else {
                            value[k] = 1 - 2 * (1 - rgb1[k]) * (1 - background[k]);//screen
                        }
                    }
                    value = value * opacity[0] + rgb2 * (1 - opacity[0]);
                    return value;
        }
        else if(compmode == std::string("SoftLight")) {
                    vec3f value;
                    for (int k = 0; k < 3; k++) {
                        if (rgb1[k] < 0.5) {
                            value[k] = 2 * rgb1[k] * background[k] + background[k] * background[k] * (1 - 2 * rgb1[k]);
                        } else {
                            value[k] = 2 * background[k] * (1 - rgb1[k]) + sqrt(background[k]) * (2 * rgb1[k] - 1);
                        }
                    }
                    /*for (int k = 0; k < 3; k++) {     Nuke method
                        if (rgb1[k] * rgb2[k] < 1) {
                            value[k] = rgb2[k] * (2 * rgb1[k] + (rgb2[k] * (1-rgb1[k] * rgb2[k])));
                        } else {
                            value[k] = 2 * rgb2[k] * rgb1[k];
                        }
                    }*/
                    value = value * opacity[0] + rgb2 * (1 - opacity[0]);
                    return value;
        }
        else if(compmode == std::string("Divide")) {
                    vec3f value;
                    for (int k = 0; k < 3; k++) {
                        if (rgb1[k] == 0) {
                            value[k] = 1;
                        } else {
                            value[k] = background[k] / rgb1[k];
                        }
                    }
                    value = value * opacity[0] + rgb2 * (1 - opacity[0]);
                    return value;
        }
        return zeno::vec3f(0);
}

struct Blend: INode {//optimize
    virtual void apply() override {//TODO:: add blend scope RGBA and Premultiplied / Alpha Blending(https://github.com/jamieowen/glsl-blend/issues/6)
        auto blend = get_input_Geometry("Foreground");
        auto base = get_input_Geometry("Background");
        auto maskopacity = get_input2_float("Mask Opacity");

        auto compmode = zsString2Std(get_input2_string("Blending Mode"));
        //auto alphablend = get_input2_string("Alpha Blending");
        auto alphamode = zsString2Std(get_input2_string("Alpha Mode"));
        auto opacity1 = get_input2_float("Foreground Opacity");
        auto opacity2 = get_input2_float("Background Opacity");

        auto ud1 = base->userData();
        int w1 = ud1->get_int("w");
        int h1 = ud1->get_int("h");
        int imagesize = w1 * h1;
        auto mask = std::make_shared<PrimitiveObject>();
        if(has_input("Mask")) {
            auto maskgeo = get_input_Geometry("Mask");
            mask = maskgeo->toPrimitiveObject();
        }
        else {
            mask->verts.resize(w1*h1);
            mask->userData()->set_int("isImage", 1);
            mask->userData()->set_int("w", w1);
            mask->userData()->set_int("h", h1);
            for (int i = 0; i < imagesize; i++) {
                    mask->verts[i] = {maskopacity,maskopacity,maskopacity};
            }
        }

        auto image2 = create_GeometryObject(zeno::Topo_IndiceMesh, true, imagesize, 0);
        image2->userData()->set_int("isImage", 1);
        image2->userData()->set_int("w", w1);
        image2->userData()->set_int("h", h1);
        bool alphaoutput =  blend->has_point_attr("alpha") || base->has_point_attr("alpha");
        image2->create_point_attr("alpha", 0.f);
        auto &image2alpha = image2->get_float_attr(ATTR_POINT, "alpha");
        const auto &blendalpha = blend->has_point_attr("alpha") ? blend->get_float_attr(zeno::ATTR_POINT, "alpha")
            : std::vector<float>(imagesize, 1.0f);
        const auto &basealpha = base->has_point_attr("alpha") ? base->get_float_attr(zeno::ATTR_POINT, "alpha")
            : std::vector<float>(imagesize, 1.0f);
        const auto& blendpos = blend->points_pos();
        const auto& basepos = base->points_pos();

#pragma omp parallel for
            for (int i = 0; i < imagesize; i++) {
                vec3f foreground = blendpos[i] * opacity1;
                vec3f rgb2 = basepos[i];
                vec3f background = rgb2 * opacity2;
                vec3f opacity = zeno::clamp(mask->verts[i] * maskopacity, 0, 1);
                float alpha1 = zeno::clamp(blendalpha[i] * opacity1, 0, 1);
                float alpha2 = zeno::clamp(basealpha[i] * opacity2, 0, 1);
                if(compmode == "Overlay" || compmode == "SoftLight" || compmode == "Divide"){
                    vec3f c = BlendModeV(alpha1, alpha2, foreground, rgb2, background, opacity, compmode);
                    image2->set_pos(i, c);
                }
                else{
                    vec3f c = BlendMode<zeno::vec3f>(alpha1, alpha2, foreground, rgb2, background, opacity, compmode);
                    image2->set_pos(i, c);
                }
            }
            if(alphaoutput) {//如果两个输入 其中一个没有alpha  对于rgb和alpha  alpha的默认值不一样 前者为1 后者为0？
                //std::string alphablendmode = alphamode == "SameWithBlend" ? compmode : alphamode;
#pragma omp parallel for
                for (int i = 0; i < imagesize; i++) {
                vec3f opacity = zeno::clamp(mask->verts[i] * maskopacity, 0, 1);
                float alpha = BlendMode<float>((blendalpha[i] * opacity1), basealpha[i],
                (blendalpha[i] * opacity1), basealpha[i], (basealpha[i] * opacity2), opacity, alphamode);
                image2alpha[i] = alpha;
                }
            }
            set_output("image", std::move(image2));
    }
};

ZENDEFNODE(Blend, {
    {
        {gParamType_Geometry, "Foreground"},
        {gParamType_Geometry, "Background"},
        {gParamType_Geometry, "Mask"},
        {"enum Over Copy Under Atop In Out Screen Add Subtract Multiply Max(Lighten) Min(Darken) Average Difference Overlay SoftLight Divide Xor", "Blending Mode", "Over"},
        //{"enum IgnoreAlpha SourceAlpha", "Alpha Blending", "Ignore Alpha"}, SUBSTANCE DESIGNER ALPHA MODE
        //{"enum SameWithBlend Over Under Atop In Out Screen Add Subtract Multiply Max(Lighten) Min(Darken) Average Difference Xor", "Alpha Mode", "SameWithBlend"},
        {"enum Over Under Atop In Out Screen Add Subtract Multiply Max(Lighten) Min(Darken) Average Difference Xor", "Alpha Mode", "Over"},
        {gParamType_Float, "Mask Opacity", "1"},
        {gParamType_Float, "Foreground Opacity", "1"},
        {gParamType_Float, "Background Opacity", "1"},
    },
    {
        {gParamType_Geometry, "image"}
    },
    {},
    { "comp" },
});

// 自定义卷积核
std::vector<std::vector<float>> createKernel(float blurValue,
                                             float l_blurValue, float r_blurValue,
                                             float t_blurValue, float b_blurValue,
                                             float lt_blurValue, float rt_blurValue,
                                             float lb_blurValue, float rb_blurValue) {
    std::vector<std::vector<float>> kernel;
    kernel = {{lt_blurValue, t_blurValue, rt_blurValue},
              {l_blurValue, blurValue, r_blurValue},
              {lb_blurValue, b_blurValue, rb_blurValue}};
    return kernel;
}

struct CompBlur : INode {//TODO::delete
    virtual void apply() override {
        auto image = get_input_Geometry("image");
        auto s = get_input2_int("strength");
        auto ktop = get_input2_vec3f("kerneltop");
        auto kmid = get_input2_vec3f("kernelmid");
        auto kbot = get_input2_vec3f("kernelbot");
        auto ud = image->userData();
        int w = ud->get_int("w");
        int h = ud->get_int("h");
        auto blurredImage = create_GeometryObject(zeno::Topo_IndiceMesh, true, w*h, 0);
        blurredImage->userData()->set_int("h", h);
        blurredImage->userData()->set_int("w", w);
        blurredImage->userData()->set_int("isImage", 1);
        if(image->has_point_attr("alpha")){
            blurredImage->create_point_attr("alpha", 0.f);
            blurredImage->set_point_attr("alpha", image->get_float_attr(ATTR_POINT, "alpha"));
        }
        std::vector<std::vector<float>>k = createKernel(kmid[1],kmid[0],kmid[2],ktop[1],kbot[1],ktop[0],ktop[2],kbot[0],kbot[2]);
// 计算卷积核的中心坐标
        int anchorX = 3 / 2;
        int anchorY = 3 / 2;
        const auto& imageverts = image->points_pos();
        for (int iter = 0; iter < s; iter++) {
#pragma omp parallel for
            // 对每个像素进行卷积操作
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    float sum0 = 0.0f;
                    float sum1 = 0.0f;
                    float sum2 = 0.0f;
                    if (x == 0 || x == w - 1 || y == 0 || y == h - 1) {
                        sum0 = imageverts[y * w + x][0];
                        sum1 = imageverts[y * w + x][1];
                        sum2 = imageverts[y * w + x][2];
                    } 
                    else
                    {
                        for (int i = 0; i < 3; i++) {
                            for (int j = 0; j < 3; j++) {
                                int kernelX = x + j - anchorX;
                                int kernelY = y + i - anchorY;
                                sum0 += imageverts[kernelY * w + kernelX][0] * k[i][j];
                                sum1 += imageverts[kernelY * w + kernelX][1] * k[i][j];
                                sum2 += imageverts[kernelY * w + kernelX][2] * k[i][j];
                            }
                        }
                    }
                    blurredImage->set_pos(y * w + x, zeno::vec3f{static_cast<float>(sum0),
                                                      static_cast<float>(sum1),
                                                      static_cast<float>(sum2)});
                }
            }
        }
        set_output("image", std::move(blurredImage));
    }
};

ZENDEFNODE(CompBlur, {
    {
        {gParamType_Geometry, "image"},
        {gParamType_Int, "strength", "5"},
        {gParamType_Vec3f, "kerneltop", "0.075,0.124,0.075"},
        {gParamType_Vec3f, "kernelmid", "0.124,0.204,0.124"},
        {gParamType_Vec3f, "kernelbot", "0.075,0.124,0.075"},
    },
    {
        {gParamType_Geometry, "image"}
    },
    {},
    { "deprecated" },
});

struct ImageExtractChannel : INode {
    virtual void apply() override {
        auto image = get_input_Geometry("image");
        auto channel = get_input2_string("channel");
        auto ud1 = image->userData();
        int w = ud1->get_int("w");
        int h = ud1->get_int("h");
        int N = image->npoints();
        auto image2 = create_GeometryObject(zeno::Topo_IndiceMesh, true, N, 0);
        image2->userData()->set_int("isImage", 1);
        image2->userData()->set_int("w", w);
        image2->userData()->set_int("h", h);
        const auto& imageverts = image->points_pos();

        if(channel == "R") {
            for (auto i = 0; i < N; i++) {
                image2->set_pos(i, vec3f(imageverts[i][0]));
            }
        }
        else if(channel == "G") {
            for (auto i = 0; i < N; i++) {
                image2->set_pos(i, vec3f(imageverts[i][1]));
            }
        }
        else if(channel == "B") {
            for (auto i = 0; i < N; i++) {
                image2->set_pos(i, vec3f(imageverts[i][2]));
            }
        }
        else if(channel == "A") {
            if (image->has_point_attr("alpha")) {
                auto &attr = image->get_float_attr(ATTR_POINT, "alpha");
                for(int i = 0; i < w * h; i++){
                    image2->set_pos(i, vec3f(attr[i]));
                }
            }
            else{
                throw zeno::makeError("image have no alpha channel");
            }
        }
        set_output("image", std::move(image2));
    }
};
ZENDEFNODE(ImageExtractChannel, {
    {
        {gParamType_Geometry, "image"},
        {"enum R G B A", "channel", "R"},
    },
    {
        {gParamType_Geometry, "image"}
    },
    {},
    { "image" },
});

/* 导入地形网格的属性，可能会有多个属性。它将地形的属性转换为图
像，每个属性对应一个图层。 */

struct CompImport : INode {
    virtual void apply() override {
        auto prim = get_input_Geometry("prim");
        auto ud = prim->userData();
        int nx = ud->has("nx") ? ud->get_int("nx") : ud->get_int("w");
        int ny = ud->has("ny") ? ud->get_int("ny") : ud->get_int("h");
        auto attrName = get_input2_string("attrName");
        auto remapRange = toVec2f(get_input2_vec2f("RemapRange"));
        auto remap = get_input2_bool("Remap");
        auto image = std::make_unique<PrimitiveObject>();
        auto attributesType = get_input2_string("AttributesType");

        image->resize(nx * ny);
        image->userData()->set_int("isImage", 1);
        image->userData()->set_int("w", nx);
        image->userData()->set_int("h", ny);

        if (attributesType == "float") {
            if (!prim->has_point_attr(attrName)) {
                throw makeError<UnimplError>("No such attribute in prim");
                return;
            }
            auto& attr = prim->get_float_attr(ATTR_POINT, attrName);
            auto minresult = zeno::parallel_reduce_array<float>(attr.size(), attr[0], [&](size_t i) -> float { return attr[i]; },
                [&](float i, float j) -> float { return zeno::min(i, j); });
            auto maxresult = zeno::parallel_reduce_array<float>(attr.size(), attr[0], [&](size_t i) -> float { return attr[i]; },
                [&](float i, float j) -> float { return zeno::max(i, j); });

            if (remap) {
                for (auto i = 0; i < nx * ny; i++) {
                    auto v = attr[i];
                    v = (v - minresult) / (maxresult - minresult);//remap to 0-1
                    v = v * (remapRange[1] - remapRange[0]) + remapRange[0];
                    image->verts[i] = vec3f(v);
                }
            }
            else {
                for (auto i = 0; i < nx * ny; i++) {
                    const auto v = attr[i];
                    image->verts[i] = vec3f(v);
                }
            }
        }
        else if (attributesType == "vec3f") {
            //TODO
        }
        auto geo = create_GeometryObject(image.get());
        set_output("image", std::move(geo));
    }
};

ZENDEFNODE(CompImport, {
    {
        {gParamType_Geometry, "prim"},
        {gParamType_String, "attrName", ""},
        {gParamType_Bool, "Remap", "0"},
        {gParamType_Vec2f, "RemapRange", "0, 1"},
        {"enum float vec3f", "AttributesType", "float"},
    },
    {
        {gParamType_Geometry, "image"},
    },
    {},
    { "comp" },
});
//TODO::Channel shuffle、RGBA Shuffle

}
}