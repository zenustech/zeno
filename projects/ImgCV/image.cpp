#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <zeno/zeno.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/NumericObject.h>
#include "tinyexr.h"
#include "zeno/utils/string.h"
#include <zeno/utils/scope_exit.h>
#include <stdexcept>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace zeno {

namespace {

static void RGBtoHSV(float r, float g, float b, float &h, float &s, float &v) {
    float rd = r;
    float gd = g;
    float bd = b;
    float cmax = fmax(rd, fmax(gd, bd));
    float cmin = fmin(rd, fmin(gd, bd));
    float delta = cmax - cmin;

    if (delta != 0) {
        if (cmax == rd) {
            h = fmod((gd - bd) / delta, 6.0);
        } else if (cmax == gd) {
            h = (bd - rd) / delta + 2.0;
        } else if (cmax == bd) {
            h = (rd - gd) / delta + 4.0;
        }
        h *= 60.0;
        if (h < 0) {
            h += 360.0;
        }
    }
    s = (cmax != 0) ? delta / cmax : 0.0;
    v = cmax;
}

static void HSVtoRGB(float h, float s, float v, float &r, float &g, float &b)
{
    int i;
    float f, p, q, t;
    if( s == 0 ) {
        // achromatic (grey)
        r = g = b = v;
        return;
    }
    h /= 60;            // sector 0 to 5
    i = floor( h );
    f = h - i;          // factorial part of h
    p = v * ( 1 - s );
    q = v * ( 1 - s * f );
    t = v * ( 1 - s * ( 1 - f ) );
    switch( i ) {
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        default:        // case 5:
            r = v;
            g = p;
            b = q;
            break;
    }
}
static void sobel(std::shared_ptr<PrimitiveObject> & grayImage, int width, int height, std::vector<float>& dx, std::vector<float>& dy)
{
    dx.resize(width * height);
    dy.resize(width * height);
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float gx = -grayImage->verts[(y - 1) * width + x - 1][0] + grayImage->verts[(y - 1) * width + x + 1][0]
                       - 2.0f * grayImage->verts[y * width + x - 1][0] + 2.0f * grayImage->verts[y * width + x + 1][0]
                       - grayImage->verts[(y + 1) * width + x - 1][0] + grayImage->verts[(y + 1) * width + x + 1][0];
            float gy = grayImage->verts[(y - 1) * width + x - 1][0] + 2.0f * grayImage->verts[(y - 1) * width + x][0] +
                       grayImage->verts[(y - 1) * width + x + 1][0]
                       - grayImage->verts[(y + 1) * width + x - 1][0] - 2.0f * grayImage->verts[(y + 1) * width + x][0] -
                       grayImage->verts[(y + 1) * width + x + 1][0];

            dx[y * width + x] = gx;
            dy[y * width + x] = gy;
        }
    }
}
static void sobel2(std::shared_ptr<PrimitiveObject> & src, std::shared_ptr<PrimitiveObject> & dst, int width, int height, int threshold) {
    std::vector<int> gx(width * height);
    std::vector<int> gy(width * height);
    dst->verts.resize(width * height);

    // Calculate gradients
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            gx[idx] = (-1 * src->verts[(y - 1) * width + x - 1][0] - 2 * src->verts[y * width + x - 1][0] - 1 * src->verts[(y + 1) * width + x - 1][0] +
                       1 * src->verts[(y - 1) * width + x + 1][0] + 2 * src->verts[y * width + x + 1][0] + 1 * src->verts[(y + 1) * width + x + 1][0]);
            gy[idx] = (-1 * src->verts[(y - 1) * width + x - 1][0] - 2 * src->verts[(y - 1) * width + x][0] - 1 * src->verts[(y - 1) * width + x + 1][0] +
                       1 * src->verts[(y + 1) * width + x - 1][0] + 2 * src->verts[(y + 1) * width + x][0] + 1 * src->verts[(y + 1) * width + x + 1][0]);
        }
    }
    // Calculate gradient magnitude
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;

            // Calculate gradient magnitude
            int mag = std::sqrt(gx[idx] * gx[idx] + gy[idx] * gy[idx]);

            // Apply thresholding
            float g = mag < threshold ? 0: 1;
            dst->verts[idx] = {g,g,g};
        }
    }
}

static void scharr2(std::shared_ptr<PrimitiveObject> & src, std::shared_ptr<PrimitiveObject> & dst, int width, int height,int threshold) {
    std::vector<int> gx(width * height);
    std::vector<int> gy(width * height);
    dst->verts.resize(width * height);

    // Calculate gradients
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            gx[idx] = (-3 * src->verts[(y - 1) * width + x - 1][0] - 10 * src->verts[y * width + x - 1][0] - 3 * src->verts[(y + 1) * width + x - 1][0] +
                       3 * src->verts[(y - 1) * width + x + 1][0] + 10 * src->verts[y * width + x + 1][0] + 3 * src->verts[(y + 1) * width + x + 1][0]);
            gy[idx] = (-3 * src->verts[(y - 1) * width + x - 1][0] - 10 * src->verts[(y - 1) * width + x][0] - 3 * src->verts[(y - 1) * width + x + 1][0] +
                       3 * src->verts[(y + 1) * width + x - 1][0] + 10 * src->verts[(y + 1) * width + x][0] + 3 * src->verts[(y + 1) * width + x + 1][0]);
        }
    }
    // Calculate gradient magnitude
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;

            // Calculate gradient magnitude
            int mag = std::sqrt(gx[idx] * gx[idx] + gy[idx] * gy[idx]);
            // Apply threshold
            if (mag * 255 > threshold) {
                // Set to white
                dst->verts[idx] = { 1, 1, 1 };
            }
            else {
                // Set to black
                dst->verts[idx] = {0, 0, 0};
            }
            // Clamp to [0, 255] and store in output image
            float g = std::min(1, std::max(0, mag));
            dst->verts[idx] = {g,g,g};
        }
    }
}
// 计算法向量
static void normalMap(std::shared_ptr<PrimitiveObject>& grayImage, int width, int height, std::vector<float>& normal)
{
    std::vector<float> dx, dy;
    sobel(grayImage, width, height, dx, dy);
    normal.resize(width * height * 3);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int i = y * width + x;
            float gx = dx[i];
            float gy = dy[i];

            float normalX = -gx;
            float normalY = -gy;
            float normalZ = 1.0f;

            float length = sqrt(normalX * normalX + normalY * normalY + normalZ * normalZ);
            normalX /= length;
            normalY /= length;
            normalZ /= length;

            normal[i * 3 + 0] = normalX;
            normal[i * 3 + 1] = normalY;
            normal[i * 3 + 2] = normalZ;
        }
    }
}

struct Composite: INode {
    virtual void apply() override {
        auto image1 = get_input2<PrimitiveObject>("Foreground");
        auto image2 = get_input2<PrimitiveObject>("Background");
        auto compmode = get_input2<std::string>("compmode");
        auto &ud1 = image1->userData();
        int w1 = ud1.get2<int>("w");
        int h1 = ud1.get2<int>("h");
        auto &ud2 = image2->userData();
        int w2 = ud2.get2<int>("w");
        int h2 = ud2.get2<int>("h");
        auto A1 = std::make_shared<PrimitiveObject>();
        A1->verts.resize(image1->size());
        A1->verts.add_attr<float>("alpha");
        for(int i = 0;i < w1 * h1;i++){
            A1->verts.attr<float>("alpha")[i] = 1.0;
        }
        auto A2 = std::make_shared<PrimitiveObject>();
        A2->verts.resize(image2->size());
        A2->verts.add_attr<float>("alpha");
        for(int i = 0;i < w2 * h2;i++){
            A2->verts.attr<float>("alpha")[i] = 1.0;
        }
        std::vector<float> &alpha1 = A1->verts.attr<float>("alpha");
        if(image1->verts.has_attr("alpha")){
            alpha1 = image1->verts.attr<float>("alpha");
        }
        if(has_input("Mask1")) {
            auto Mask1 = get_input2<PrimitiveObject>("Mask1");
            Mask1->verts.add_attr<float>("alpha");
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    Mask1->verts.attr<float>("alpha")[i * w1 + j] = Mask1->verts[i * w1 + j][0];
                }
            }
            alpha1 = Mask1->verts.attr<float>("alpha");
        }
        std::vector<float> &alpha2 = A2->verts.attr<float>("alpha");
        if(image2->verts.has_attr("alpha")){
            alpha2 = image2->verts.attr<float>("alpha");
        }
        if(has_input("Mask2")) {
            auto Mask2 = get_input2<PrimitiveObject>("Mask2");
            Mask2->verts.add_attr<float>("alpha");
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    Mask2->verts.attr<float>("alpha")[i * w1 + j] = Mask2->verts[i * w1 + j][0];
                }
            }
            alpha2 = Mask2->verts.attr<float>("alpha");
        }

        if (compmode == "Over") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = rgb1 * l1 + rgb2 * (l2 - ((l1 != 0) && (l2 != 0) ? l2 : 0));
                }
            }
        }
        if (compmode == "Under") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = rgb2 * l2 + rgb1 * (l1 - ((l1 != 0) && (l2 != 0) ? l1 : 0));
                }
            }
        }
        if (compmode == "Atop") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] =
                            rgb1 * ((l1 != 0) && (l2 != 0) ? l1 : 0) + rgb2 * ((l1 == 0) && (l2 != 0) ? l2 : 0);
                }
            }
        }
        if (compmode == "Inside") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = rgb1 * ((l1 != 0) && (l2 != 0) ? l1 : 0);
                }
            }
        }
        if (compmode == "Outside") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = rgb1 * ((l1 != 0) && (l2 == 0) ? l1 : 0);
                }
            }
        }
        if(compmode == "Screen"){
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float var = (image1->verts[i * w1 + j][0]+image1->verts[i * w1 + j][1]+image1->verts[i * w1 + j][2])/3;
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = rgb2 * l2 + rgb2 *((l1!=0 && l2!=0)? var: 0);
                }
            }
        }
        if (compmode == "Add") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = rgb1 * l1 + rgb2 * l2;
                }
            }
        }
        if (compmode == "Subtract") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = rgb1 * l1 - rgb2 * l2;
                }
            }
        }
        if (compmode == "Multiply") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = rgb1 * l1 * rgb2 * l2;
                }
            }
        }
        if (compmode == "Divide") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = rgb1 * l1 / (rgb2 * l2);
                }
            }
        }
        if (compmode == "Diff") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = abs(rgb1 * l1 - (rgb2 * l2));
                }
            }
        }
        if (compmode == "Min") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = l1 <= l2 ? rgb1 * l1 : rgb2 * l2;
                }
            }
        }
        if (compmode == "Max") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = l1 >= l2 ? rgb1 * l1 : rgb2 * l2;
                }
            }
        }
        if (compmode == "Average") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    vec3f rgb3 = (rgb1+rgb2)/2;
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = rgb3 * (l1+l2);
                }
            }
        }
        if (compmode == "Xor") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    vec3f rgb3 = {0, 0, 0};
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = (((l1 != 0) && (l2 != 0)) ? rgb3 : rgb1 * l1 + rgb2 * l2);
                }
            }
        }
        if (compmode == "Alpha") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    vec3f rgb3 = {1,1,1};
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = rgb3 * ((l1 != 0) || (l2 != 0) ? 1 : 0);
                }
            }
        }
        if (compmode == "!Alpha") {
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    vec3f rgb3 = {1,1,1};
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    image1->verts[i * w1 + j] = rgb3 * ((l1 != 0) || (l2 != 0) ? 0 : 1);
                }
            }
        }
        set_output("image", image1);
    }
};

ZENDEFNODE(Composite, {
    {
        {"Foreground"},
        {"Background"},
        {"Mask1"},
        {"Mask2"},
        {"enum Over Under Atop Inside Outside Screen Add Subtract Multiply Divide Diff Min Max Average Xor Alpha !Alpha", "compmode", "Over"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

//replaced by Composite
/*
struct CompositeCV: INode {
    virtual void apply() override {
        auto image1 = get_input<PrimitiveObject>("Foreground");
        auto image2 = get_input<PrimitiveObject>("Background");
        auto mode = get_input2<std::string>("mode");
        auto Alpha1 = get_input2<float>("Alpha1");
        auto Alpha2 = get_input2<float>("Alpha2");
        auto &a1 = image1->verts.attr<float>("alpha");
        auto &a2 = image2->verts.attr<float>("alpha");

        auto &ud1 = image1->userData();
        int w1 = ud1.get2<int>("w");
        int h1 = ud1.get2<int>("h");
        auto &ud2 = image2->userData();
        int w2 = ud2.get2<int>("w");
        int h2 = ud2.get2<int>("h");

        cv::Mat imagecvin1(h1, w1, CV_32FC3);
        cv::Mat imagecvin2(h2, w2, CV_32FC3);
        cv::Mat imagecvadd(h1, w1, CV_32FC3);
        cv::Mat imagecvsub(h1, w1, CV_32FC3);
        cv::Mat imagecvout(h1, w1, CV_32FC3);

        for (int i = 0; i < h1; i++) {
            for (int j = 0; j < w1; j++) {
                vec3f rgb = image1->verts[i * w1 + j];
                imagecvin1.at<cv::Vec3f>(i, j) = {rgb[0], rgb[1], rgb[2]};
            }
        }
        for (int i = 0; i < h2; i++) {
            for (int j = 0; j < w2; j++) {
                vec3f rgb = image2->verts[i * w2 + j];
                imagecvin2.at<cv::Vec3f>(i, j) = {rgb[0], rgb[1], rgb[2]};
            }
        }
        cv::resize(imagecvin2, imagecvin2,imagecvin1.size());
        if(mode == "Add"){
            cv::addWeighted(imagecvin1, Alpha1, imagecvin2, Alpha2, 0, imagecvout);
        }
        if(mode == "Subtract"){
            cv::subtract(imagecvin1*Alpha1, imagecvin2*Alpha2, imagecvout);
        }
        if(mode == "Multiply"){
            cv::multiply(imagecvin1*Alpha1, imagecvin2*Alpha2, imagecvout);
        }
        if(mode == "Divide"){
            cv::divide(imagecvin1*Alpha1, imagecvin2*Alpha2, imagecvout, 1,  -1);
        }
        if(mode == "Diff"){
            cv::absdiff(imagecvin1*Alpha1, imagecvin2*Alpha2, imagecvout);
        }

        for (int i = 0; i < h1; i++) {
            for (int j = 0; j < w1; j++) {
                cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
                image1->verts[i * w1 + j] = {rgb[0], rgb[1], rgb[2]};
            }
        }
        set_output("image", image1);
    }
};

ZENDEFNODE(CompositeCV, {
    {
        {"Foreground"},
        {"Background"},
        {"enum Add Subtract Multiply Divide Diff", "mode", "Add"},
        {"float", "Alpha1", "1"},
        {"float", "Alpha2", "1"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});
*/

struct ImageRGB2HSV : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        float H = 0, S = 0, V = 0;
        for (auto i = 0; i < image->verts.size(); i++){
            float R = image->verts[i][0];
            float G = image->verts[i][1];
            float B = image->verts[i][2];
            zeno::RGBtoHSV(R, G, B, H, S, V);
            image->verts[i][0]= H;
            image->verts[i][1]= S;
            image->verts[i][2]= V;
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageRGB2HSV, {
    {
        {"image"},
    },
    {
        {"image"},
    },
    {},
    { "comp" },
});

struct ImageHSV2RGB : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        float R = 0, G = 0, B = 0;
        for (auto i = 0; i < image->verts.size(); i++){
            float H = image->verts[i][0];
            float S = image->verts[i][1];
            float V = image->verts[i][2];
            zeno::HSVtoRGB(H, S, V, R, G, B);
            image->verts[i][0]= R ;
            image->verts[i][1]= G ;
            image->verts[i][2]= B ;
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageHSV2RGB, {
    {
        {"image"},
    },
    {
        {"image"},
    },
    {},
    { "comp" },
});

struct ImageEditRGB : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto RGB = get_input2<std::string>("RGB");
        auto Gray = get_input2<bool>("Gray");
        auto Invert = get_input2<bool>("Invert");
        float R = get_input2<float>("R");
        float G = get_input2<float>("G");
        float B = get_input2<float>("B");

        if(RGB == "RGB") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = R * image->verts[i][0];
                float G1 = G * image->verts[i][1];
                float B1 = B * image->verts[i][2];
                image->verts[i][0] = R1 ;
                image->verts[i][1] = G1 ;
                image->verts[i][2] = B1 ;
            }
        }
        if(RGB == "R") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = R * image->verts[i][0];
                float G1 = 0;
                float B1 = 0;
                image->verts[i][0] = R1 ;
                image->verts[i][1] = G1 ;
                image->verts[i][2] = B1 ;
            }
        }
        if(RGB == "G") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = 0;
                float G1 = G * image->verts[i][1];
                float B1 = 0;
                image->verts[i][0] = R1 ;
                image->verts[i][1] = G1 ;
                image->verts[i][2] = B1 ;
            }
        }
        if(RGB == "B") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = 0;
                float G1 = 0;
                float B1 = B * image->verts[i][2];
                image->verts[i][0] = R1;
                image->verts[i][1] = G1;
                image->verts[i][2] = B1;
            }
        }
        if(Gray){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                float avr = (R + G + B)/3;
                image->verts[i][0] = avr ;
                image->verts[i][1] = avr ;
                image->verts[i][2] = avr ;
            }
        }
        if(Invert){
            for (auto i = 0; i < image->verts.size(); i++) {
                image->verts[i] = 1 - image->verts[i];
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageEditRGB, {
    {
        {"image"},
        {"enum RGB R G B", "RGB", "RGB"},
        {"float", "R", "1"},
        {"float", "G", "1"},
        {"float", "B", "1"},
        {"bool", "Gray", "0"},
        {"bool", "Invert", "0"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

struct ImageEditHSV : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        float H = 0, S = 0, V = 0;
        auto Hue = get_input2<std::string>("Hue");
        float Hi = get_input2<float>("H");
        float Si = get_input2<float>("S");
        float Vi = get_input2<float>("V");
        if(Hue == "default"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(Vi-1);
                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "edit"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = Hi;
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(Vi-1);
                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "red"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = 0;
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(Vi-1);
                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "orange"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = 30;
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(Vi-1);
                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "yellow"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = 60;
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(Vi-1);
                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "green"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = 120;
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(Vi-1);
                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "cyan"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = 180;
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(Vi-1);
                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "blue"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = 240;
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(Vi-1);
                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "purple"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = 300;
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(Vi-1);
                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageEditHSV, {
    {
        {"image"},
        {"enum default edit red orange yellow green cyan blue purple ", "Hue", "edit"},
        {"float", "H", "1"},
        {"float", "S", "1"},
        {"float", "V", "1"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

struct ImageEdit: INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto RGB = get_input2<std::string>("RGB");
        auto Gray = get_input2<bool>("Gray");
        auto Invert = get_input2<bool>("Invert");

        float R = get_input2<float>("R");
        float G = get_input2<float>("G");
        float B = get_input2<float>("B");
        float L = get_input2<float>("Luminace");
        float ContrastRatio = get_input2<float>("ContrastRatio");
        float Si = get_input2<float>("Saturation");
        float H = 0, S = 0, V = 0;

        if(RGB == "RGB") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = image->verts[i][0];
                float G1 = image->verts[i][1];
                float B1 = image->verts[i][2];
                R1 *= R;
                G1 *= G;
                B1 *= B;
                zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(L-1);
                zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                image->verts[i][0] = R1;
                image->verts[i][1] = G1;
                image->verts[i][2] = B1;
            }
        }
        if(RGB == "R") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = R * image->verts[i][0];
                float G1 = 0;
                float B1 = 0;
                R1 *= R;
                G1 *= G;
                B1 *= B;
                zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(L-1);
                zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                image->verts[i][0] = R1 ;
                image->verts[i][1] = G1 ;
                image->verts[i][2] = B1 ;
            }
        }
        if(RGB == "G") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = 0;
                float G1 = G * image->verts[i][1];
                float B1 = 0;
                R1 *= R;
                G1 *= G;
                B1 *= B;
                zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(L-1);
                zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                image->verts[i][0] = R1 ;
                image->verts[i][1] = G1 ;
                image->verts[i][2] = B1 ;
            }
        }
        if(RGB == "B") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = 0;
                float G1 = 0;
                float B1 = B * image->verts[i][2];
                R1 *= R;
                G1 *= G;
                B1 *= B;
                zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(L-1);
                zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                image->verts[i][0] = R1;
                image->verts[i][1] = G1;
                image->verts[i][2] = B1;
            }
        }
        for (auto i = 0; i < image->verts.size(); i++) {
            image->verts[i] = image->verts[i] + (image->verts[i]-0.5) * (ContrastRatio-1);
        }
        if(Gray){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                float avr = (R + G + B)/3;
                image->verts[i][0] = avr ;
                image->verts[i][1] = avr ;
                image->verts[i][2] = avr ;
            }
        }
        if(Invert){
            for (auto i = 0; i < image->verts.size(); i++) {
                image->verts[i] = 1 - image->verts[i];
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageEdit, {
    {
        {"image"},
        {"enum RGB R G B", "RGB", "RGB"},
        {"float", "R", "1"},
        {"float", "G", "1"},
        {"float", "B", "1"},
        {"float", "Saturation", "1"},
        {"float", "Luminace", "1"},
        {"float", "ContrastRatio", "1"},
        {"bool", "Gray", "0"},
        {"bool", "Invert", "0"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

// 高斯函数
float gaussian(float x, float sigma) {
    return exp(-(x * x) / (2 * sigma * sigma));
}
// 高斯滤波函数
void gaussian_filter(std::shared_ptr<PrimitiveObject> &image, std::shared_ptr<PrimitiveObject> &imagetmp, int width, int height, int sigma) {
    // 计算高斯核大小
    int size = (int)(2 * sigma + 1);
    if (size % 2 == 0) {
        size++;
    }
    // 创建高斯核
    float* kernel = new float[size];
    float sum = 0.0;
    int mid = size / 2;
    for (int i = 0; i < size; i++) {
        kernel[i] = gaussian(i - mid, sigma);
        sum += kernel[i];
    }
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
    // 对每个像素进行卷积操作
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum0 = 0.0, sum1 = 0.0, sum2 = 0.0;
            for (int i = -mid; i <= mid; i++) {
                int nx = x + i;
                if (nx < 0 || nx >= width) {
                    continue;
                }
                sum0 += kernel[i + mid] * image->verts[y * width + nx][0];
                sum1 += kernel[i + mid] * image->verts[y * width + nx][1];
                sum2 += kernel[i + mid] * image->verts[y * width + nx][2];
            }
            imagetmp->verts[y * width + x] = {sum0,sum1,sum2};
        }
    }
    image = imagetmp;
    // 释放内存
    delete[] kernel;
}
//MedianBlur
// 定义一个结构体，用于存储像素点的信息
struct Pixel {
    int x;
    int y;
    int value;
};
// MedianBlur函数，实现中值滤波操作
void MedianBlur(std::shared_ptr<PrimitiveObject> &image, std::shared_ptr<PrimitiveObject> &imagetmp, int width, int height, int kernel_size) {
    // 定义一个vector，用于存储周围像素的值
    using kernel = std::tuple<float, float, float>;
    kernel n = {0, 0, 0};
    std::vector<kernel> kernel_values(kernel_size * kernel_size);
    // 遍历图像中的每个像素点
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // 获取当前像素点的值
            int current_value0 = image->verts[y * width + x][0];
            int current_value1 = image->verts[y * width + x][1];
            int current_value2 = image->verts[y * width + x][2];
            // 遍历周围像素，获取像素值和坐标信息
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int px = x - kernel_size / 2 + kx;
                    int py = y - kernel_size / 2 + ky;
                    // 判断像素是否越界，如果越界则使用当前像素值作为周围像素值
                    if (px < 0 || px >= width || py < 0 || py >= height) {
                        kernel_values[ky * kernel_size + kx] = {current_value0,current_value1,current_value2};
                    }
                    else {
                        kernel_values[ky * kernel_size + kx] = {image->verts[py * width + px][0],image->verts[py * width + px][1],image->verts[py * width + px][2]};
                    }
                }
            }
            // 对周围像素的值进行排序，并取中间值作为新的像素值
            std::sort(kernel_values.begin(), kernel_values.end());
            float new_value0 = std::get<0>(kernel_values[kernel_size * kernel_size / 2]);
            float new_value1 = std::get<1>(kernel_values[kernel_size * kernel_size / 2]);
            float new_value2 = std::get<2>(kernel_values[kernel_size * kernel_size / 2]);
            // 将新的像素值赋值给输出图像
            imagetmp->verts[y * width + x] = {new_value0,new_value1,new_value2};
        }
    }
    image = imagetmp;
}


// 定义一个函数，用于计算双边权重
float bilateral(float src, float dst, float sigma_s, float sigma_r) {
    return gaussian(src - dst, sigma_s) * gaussian(abs(src - dst), sigma_r);
}

// 定义一个函数，用于对图像进行双边滤波
void bilateralFilter(std::shared_ptr<PrimitiveObject> &image, std::shared_ptr<PrimitiveObject> &imagetmp, int width, int height, float sigma_s, float sigma_r) {
    // 计算卷积核的半径
    int k = ceil(3 * sigma_s);
    // 定义一个临时数组，用于存储每个像素点的中间值
    float* tmp = new float[width * height];
    for (int i = k; i < height-k; i++) {
        for (int j = k; j < width-k; j++) {
            // 定义变量，用于存储像素值的加权平均值
            float sum0 = 0, sum1 = 0, sum2 = 0;
            // 定义变量，用于存储权重的和
            float wsum0 = 0,wsum1 = 0,wsum2 = 0;
            for (int m = -k; m <= k; m++) {
                for (int n = -k; n <= k; n++) {
                    // 计算双边权重
                    float w0 = bilateral(image->verts[i*width+j][0],image->verts[(i+m)*width+j+n][0], sigma_s, sigma_r);
                    float w1 = bilateral(image->verts[i*width+j][1],image->verts[(i+m)*width+j+n][1], sigma_s, sigma_r);
                    float w2 = bilateral(image->verts[i*width+j][2],image->verts[(i+m)*width+j+n][2], sigma_s, sigma_r);
                    // 计算加权平均值
                    sum0 += w0 * image->verts[(i+m)*width+j+n][0];
                    sum1 += w1 * image->verts[(i+m)*width+j+n][1];
                    sum2 += w2 * image->verts[(i+m)*width+j+n][2];
                    // 计算权重的和
                    wsum0 += w0;
                    wsum1 += w1;
                    wsum2 += w2;
                }
            }
            imagetmp->verts[i*width+j] = {sum0 / wsum0,sum1/ wsum1, sum2 / wsum2};   // 计算每个像素点的中间值，并将结果存储到临时数组中
        }
    }
    image = imagetmp;
}


struct ImageEditBlur : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        auto xsize = get_input2<int>("xsize");
        auto ysize = get_input2<int>("ysize");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        auto imagetmp = std::make_shared<PrimitiveObject>();
        imagetmp->resize(w * h);
        imagetmp->userData().set2("isImage", 1);
        imagetmp->userData().set2("w", w);
        imagetmp->userData().set2("h", h);

        cv::Mat imagecvin(h, w, CV_32FC3);
        cv::Mat imagecvout(h, w, CV_32FC3);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                vec3f rgb = image->verts[i * w + j];
                imagecvin.at<cv::Vec3f>(i, j) = {rgb[0], rgb[1], rgb[2]};
            }
        }
        if(mode=="Blur"){
            cv::blur(imagecvin,imagecvout,cv::Size(xsize,ysize),cv::Point(-1,-1));
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
                    image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
                }
            }
        }
        if(mode=="CVGaussianBlur"){
            if(xsize%2==0){
                xsize += 1;
            }
            cv::GaussianBlur(imagecvin,imagecvout,cv::Size(xsize,xsize),1.5);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
                    image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
                }
            }
        }
        if(mode=="CVMedianBlur"){
            if(xsize%2==0){
                xsize += 1;
            }
            cv::medianBlur(imagecvin,imagecvout,xsize);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
                    image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
                }
            }
        }
        if(mode=="CVBilateralBlur"){
            if(xsize%2==0){
                xsize += 1;
            }
            cv::bilateralFilter(imagecvin,imagecvout,xsize,75,75);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
                    image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
                }
            }
        }
        if(mode=="GaussianBlur"){
            // 进行高斯滤波
            gaussian_filter(image,imagetmp, w, h, xsize);
        }
        if(mode=="MedianBlur"){
            MedianBlur(image,imagetmp, w, h, xsize);
        }
        if(mode=="BilateralBlur"){
            bilateralFilter(image,imagetmp, w, h, xsize, ysize);
        }

        set_output("image", image);
    }
};

ZENDEFNODE(ImageEditBlur, {
    {
        {"image"},
        {"enum Blur GaussianBlur MedianBlur BilateralBlur CVGaussianBlur CVMedianBlur CVBilateralBlur", "mode", "Blur"},
        {"float", "xsize", "10"},
        {"float", "ysize", "10"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

//边缘检测
struct EdgeDetect : INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        int low_threshold = get_input2<int>("low_threshold");
        int high_threshold = get_input2<int>("high_threshold");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        auto image2 = std::make_shared<PrimitiveObject>();
        image2->resize(w * h);
        image2->userData().set2("isImage", 1);
        image2->userData().set2("w", w);
        image2->userData().set2("h", h);
        if(mode=="canny"){
            cv::Mat imagecvin(h, w, CV_8U);
            cv::Mat imagecvout(h, w, CV_8U);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    imagecvin.at<uchar>(i, j) = int(rgb[0] * 255);
                }
            }
            cv::Canny(imagecvin,imagecvout,low_threshold, high_threshold);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<uchar>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
        if(mode=="sobel"){
            std::vector<float> dx,dy;
            zeno::sobel(image, w, h, dx, dy);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    // 计算梯度幅值
                    float gradient = std::sqrt(pow(dx[i * w + j],2) + pow(dy[i * w + j],2));
                    image->verts[i * w + j] = {gradient,gradient,gradient};
                }
            }
            set_output("image", image);
        }
        if(mode=="sobel2"){
            std::vector<float> dx,dy;
            zeno::sobel2(image, image2, w , h, low_threshold);
            set_output("image", image2);
        }

        if(mode=="scharr"){
            zeno::scharr2(image, image2, w, h, low_threshold);
            set_output("image", image2);
        }

        if(mode=="laplacian"){

        }
    }
};
ZENDEFNODE(EdgeDetect, {
   {
       {"image"},
       {"enum sobel canny scharr laplacian sobel2", "mode", "sobel"},
       {"int", "low_threshold", "100"},
       {"int", "high_threshold", "150"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

struct EdgeDetect2 : INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        int threshold = ud.get2<int>("threshold");
        auto image2 = std::make_shared<PrimitiveObject>();
        image2->resize(w * h);
        image2->userData().set2("isImage", 1);
        image2->userData().set2("w", w);
        image2->userData().set2("h", h);
        if(mode=="sobel"){
            zeno::sobel2(image, image2, w, h,threshold);
            set_output("image", image2);
        }

        if(mode=="scharr"){
            zeno::scharr2(image, image2, w, h,threshold);
            set_output("image", image2);
        }

        if(mode=="laplacian"){

        }
    }
};
ZENDEFNODE(EdgeDetect2, {
    {
        {"image"},
        {"enum sobel scharr laplacian", "mode", "sobel"},
        {"int", "threshold", "150"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

struct CompExtractRGBA_gray : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto RGBA = get_input2<std::string>("RGBA");

        if(RGBA == "RGBA") {}
        if(RGBA == "R") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                image->verts[i][0] = R;
                image->verts[i][1] = R;
                image->verts[i][2] = R;
            }
        }
        if(RGBA == "G") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float G = image->verts[i][1];
                image->verts[i][0] = G;
                image->verts[i][1] = G;
                image->verts[i][2] = G;
            }
        }
        if(RGBA == "B") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float B = image->verts[i][2];
                image->verts[i][0] = B;
                image->verts[i][1] = B;
                image->verts[i][2] = B;
            }
        }
        if(RGBA == "A") {
            if (image->verts.has_attr("alpha")) {
                auto &Alpha = image->verts.attr<float>("alpha");
                for (auto i = 0; i < image->verts.size(); i++) {
                    float A = Alpha[i];
                    image->verts[i][0] = A;
                    image->verts[i][1] = A;
                    image->verts[i][2] = A;
                }
            }
        }
        set_output("image", image);
    }
};
ZENDEFNODE(CompExtractRGBA_gray, {
    {
        {"image"},
        {"enum RGBA R G B A", "RGBA", "RGBA"},
        {"int", "R", "1"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

struct CompExtractRGBA : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto RGBA = get_input2<bool>("RGBA");
        auto R = get_input2<bool>("R");
        auto G = get_input2<bool>("G");
        auto B = get_input2<bool>("B");
        auto A = get_input2<bool>("A");

        auto A1 = std::make_shared<PrimitiveObject>();
        A1->verts.resize(image->size());
        A1->verts.add_attr<float>("alpha");
        for(int i = 0;i < image->size();i++){
            A1->verts.attr<float>("alpha")[i] = 1.0;
        }
        std::vector<float> &Alpha = A1->verts.attr<float>("alpha");
        if(image->verts.has_attr("alpha")){
            Alpha = image->verts.attr<float>("alpha");
        }

        float R1=0,G1=0,B1=0;

        if(!R&&!RGBA) {
            for (auto i = 0; i < image->verts.size(); i++) {
                R1 = image->verts[i][0];
                image->verts[i][0] = R1 * ((Alpha[i] == 1) ? 0 : 1);
            }
        }

        if(!G&&!RGBA) {
            for (auto i = 0; i < image->verts.size(); i++) {
                G1 = image->verts[i][1];
                image->verts[i][1] = G1 * ((Alpha[i] == 1) ? 0 : 1);
            }
        }
        if(!B&&!RGBA) {
            for (auto i = 0; i < image->verts.size(); i++) {
                B1 = image->verts[i][2];
                image->verts[i][2] = B1 * ((Alpha[i] == 1) ? 0 : 1);
            }
        }

        if(A) {
            if (image->verts.has_attr("alpha")) {
                for (auto i = 0; i < image->verts.size(); i++) {
                    float A = Alpha[i];
                    image->verts[i][0] = A;
                    image->verts[i][1] = A;
                    image->verts[i][2] = A;
                }
            }
        }
        set_output("image", image);
    }
};
ZENDEFNODE(CompExtractRGBA, {
    {
        {"image"},
        {"bool", "RGBA", "0"},
        {"bool", "R", "0"},
        {"bool", "G", "0"},
        {"bool", "B", "0"},
        {"bool", "A", "0"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});
struct ImageInRange : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        vec3i lb = get_input2<vec3i>("low_threshold");
        vec3i ub = get_input2<vec3i>("high_threshold");

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if(((lb[0] <= 255 * image->verts[i * w + j][0]) && (255 * image->verts[i * w + j][0] <= ub[0])) &&
                ((lb[1] <= 255 * image->verts[i * w + j][1]) && (255 * image->verts[i * w + j][1] <= ub[1])) &&
                ((lb[2] <= 255 * image->verts[i * w + j][2]) && (255 * image->verts[i * w + j][2] <= ub[2]))){
                    image->verts[i * w + j] = {1,1,1};
                }
                else{
                    image->verts[i * w + j] = {0,0,0};
                }
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageInRange, {
    {
        {"image"},
        {"vec3i", "high_threshold", "255,255,255"},
        {"vec3i", "low_threshold", "0,0,0"},
    },
    {
        {"image"},
    },
    {},
    { "comp" },
});
/* 导入地形网格的属性，可能会有多个属性。它将地形的属性转换为图
像，每个属性对应一个图层。
可能需要的参数：outRemapRange，分辨率，属性名称，属性数据
类型为float32 */
struct CompImport : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto &ud = prim->userData();
        int nx = ud.get2<int>("nx");
        int ny = ud.get2<int>("ny");
        auto attrName = get_input2<std::string>("attrName");

        auto image = std::make_shared<PrimitiveObject>();
        image->resize(nx * ny);
        image->userData().set2("isImage", 1);
        image->userData().set2("w", nx);
        image->userData().set2("h", ny);

        if (prim->verts.attr_is<float>(attrName)) {
            auto &attr = prim->attr<float>(attrName);
            for (auto i = 0; i < nx * ny; i++) {
                float v = attr[i];
                image->verts[i] = {v, v, v};
            }
        }
        else if (prim->verts.attr_is<vec3f>(attrName)) {
            auto &attr = prim->attr<vec3f>(attrName);
            for (auto i = 0; i < nx * ny; i++) {
                image->verts[i] = attr[i];
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(CompImport, {
    {
        {"prim"},
        {"string", "attrName", ""},
    },
    {
        {"image"},
    },
    {},
    { "comp" },
});

/* 创建颜色图层，可能需要的参数：颜色，分辨率，图层名称 */
struct CreateColor : INode {
    virtual void apply() override {
        auto RGB = get_input2<vec3f>("RGB");
        auto x = get_input2<int>("width");
        auto y = get_input2<int>("height");

        auto image = std::make_shared<PrimitiveObject>();
        image->verts.resize(x * y);
        image->userData().set2("h", y);
        image->userData().set2("w", x);
        image->userData().set2("isImage", 1);
        for (int i = 0; i < x * y; i++){
            image->verts[i] = {RGB[0],RGB[1],RGB[2]};
        }
        set_output("image", image);
    }
};

ZENDEFNODE(CreateColor, {
    {
        {"vec3f", "RGB", "1,1,1"},
        {"int", "width", "256"},
        {"int", "height", "256"},
    },
    {
        {"image"}
        },
    {},
    { "comp" },
});

//对比度
struct ImageEditContrast : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        float ContrastRatio = get_input2<float>("ContrastRatio");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        for (auto i = 0; i < image->verts.size(); i++) {
            image->verts[i] = image->verts[i] + (image->verts[i]-0.5) * (ContrastRatio-1);
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageEditContrast, {
    {
        {"image"},
        {"float", "ContrastRatio", "1"},
    },
    {"image"},
    {},
    { "comp" },
});

//饱和度
struct ImageEditSaturation : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        float Si = get_input2<float>("Saturation");
        float H = 0, S = 0, V = 0;
        for (auto i = 0; i < image->verts.size(); i++) {
            float R = image->verts[i][0];
            float G = image->verts[i][1];
            float B = image->verts[i][2];
            zeno::RGBtoHSV(R, G, B, H, S, V);
            S = S + (S - 0.5)*(Si-1);
            zeno::HSVtoRGB(H, S, V, R, G, B);
            image->verts[i][0] = R;
            image->verts[i][1] = G;
            image->verts[i][2] = B;
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageEditSaturation, {
    {
        {"image"},
        {"float", "Saturation", "1"},
    },
    {"image"},
    {},
    { "comp" },
});

struct ImageEditInvert : INode{
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        for (auto i = 0; i < image->verts.size(); i++) {
            image->verts[i] = 1 - image->verts[i];
        }
        set_output("image", image);
    }
};
ZENDEFNODE(ImageEditInvert, {
    {
        {"image"},
    },
    {
        "image",
    },
    {},
    {"comp"},
});

/* 将灰度图像转换为法线贴图 */
struct CompNormalMap : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto strength = get_input2<float>("strength");
        auto InvertR = get_input2<bool>("InvertR");
        auto InvertG = get_input2<bool>("InvertG");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        using normal =  std::tuple<float, float, float>;
        normal n = {0, 0, 1};
        float n0 = std::get<0>(n);
        float n1 = std::get<1>(n);
        float n2 = std::get<2>(n);
        std::vector<normal> normalmap;
        normalmap.resize(image->size());
        float gx = 0;
        float gy = 0;
        float gz = 1;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int idx = i * w + j;
                if (i == 0 || i == h || j == 0 || j == w) {
                    normalmap[idx] = {0, 0, 1};
                }
            }
        }
        for (int i = 1; i < h-1; i++) {
            for (int j = 1; j < w-1; j++) {
                int idx = i * w + j;
                gx = (image->verts[idx+1][0] - image->verts[idx-1][0])/2.0f * strength;
                gy = (image->verts[idx+w][0] - image->verts[idx-w][0])/2.0f * strength;
                float len = sqrt(gx * gx + gy * gy + gz * gz);
                gx /= len;
                gy /= len;
                gz /= len;
                // 计算光照值
                if((!InvertG && !InvertR) || (InvertG && InvertR)){
                    gx = 0.5f * (gx + 1.0f) ;
                    gy = 0.5f * (-gy + 1.0f) ;
                    gz = 0.5f * (gz + 1.0f) ;
                    normalmap[i * w + j] = {gx,gy,gz};
                }
                else if((!InvertG && InvertR) || (InvertG && !InvertR)){
                    gx = 0.5f * (gx + 1.0f) ;
                    gy = 0.5f * (gy + 1.0f) ;
                    gz = 0.5f * (gz + 1.0f) ;
                    normalmap[i * w + j] = {gx,gy,gz};
                }
            }
        }
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int idx = i * w + j;
                if(!InvertG && !InvertR){
                    image->verts[i * w + j][0] = std::get<0>(normalmap[i * w + j]);
                    image->verts[i * w + j][1] = std::get<1>(normalmap[i * w + j]);
                    image->verts[i * w + j][2] = std::get<2>(normalmap[i * w + j]);
                }
                if(!InvertG && InvertR){
                    image->verts[i * w + j][0] = std::get<1>(normalmap[i * w + j]);
                    image->verts[i * w + j][1] = std::get<0>(normalmap[i * w + j]);
                    image->verts[i * w + j][2] = std::get<2>(normalmap[i * w + j]);
                }
                if(InvertG && !InvertR){
                    image->verts[i * w + j][0] = std::get<0>(normalmap[i * w + j]);
                    image->verts[i * w + j][1] = std::get<1>(normalmap[i * w + j]);
                    image->verts[i * w + j][2] = std::get<2>(normalmap[i * w + j]);
                }
                if(InvertG && InvertR){
                    image->verts[i * w + j][0] = std::get<1>(normalmap[i * w + j]);
                    image->verts[i * w + j][1] = std::get<0>(normalmap[i * w + j]);
                    image->verts[i * w + j][2] = std::get<2>(normalmap[i * w + j]);
                }
            }
        }
        set_output("image", image);
    }
};
ZENDEFNODE(CompNormalMap, {
    {
        {"image"},
        {"float", "strength", "25"},
        {"bool", "InvertR", "0"},
        {"bool", "InvertG", "0"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});


struct ImageEditGray : INode {
    void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        for (auto i = 0; i < image->verts.size(); i++) {
            vec3f rgb = image->verts[i];
            if(mode=="average"){
                float avg = (rgb[0] + rgb[1] + rgb[2]) / 3;
                image->verts[i] = {avg, avg, avg};
            }
            if(mode=="luminace"){
                float l = std::min(std::min(rgb[0], rgb[1]),rgb[2]);
                image->verts[i] = {l,l,l};
            }
        }
        set_output("image", image);
    }
};
ZENDEFNODE(ImageEditGray, {
    {
        {"image"},
        {"enum average luminace", "mode", "average"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

struct ImageGetSize : INode {
    void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        set_output2("width", w);
        set_output2("height", h);
    }
};
ZENDEFNODE(ImageGetSize, {
    {
        {"image"},
    },
    {
        {"int", "width"},
        {"int", "height"},
    },
    {},
    {"comp"},
});

static std::shared_ptr<PrimitiveObject> normal_tiling(std::shared_ptr<PrimitiveObject> &image1, std::shared_ptr<PrimitiveObject> &image2, int rows, int cols) {
    int width = image1->userData().get2<int>("w");
    int height = image1->userData().get2<int>("h");
    int new_width = width * cols;
    int new_height = height * rows;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int x = j * width;
            int y = i * height;
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int index1 = (y + h) * new_width + (x + w);
                    int index2 = h * width + w;
                    image2->verts[index1] = image1->verts[index2];
                }
            }
        }
    }
    return image2;
}

static std::shared_ptr<PrimitiveObject> mirror_tiling(std::shared_ptr<PrimitiveObject> &image1, std::shared_ptr<PrimitiveObject> &image2, int rows, int cols) {
    int width = image1->userData().get2<int>("w");
    int height = image1->userData().get2<int>("h");
    int new_width = width * cols;
    int new_height = height * rows;
    // 复制像素并进行镜像平铺
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int x = j * width;
            int y = i * height;
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    if(i%2 == 0 && j%2 == 0){
                        image2->verts[(y + h) * width * cols + (x + w)] = image1->verts[h * width + w];
                    }
                    if(i%2 == 0 && j%2 == 1){
                        image2->verts[((y + h) * width * cols + x + (width - w - 1))] = image1->verts[(h * width + w)];
                    }
                    if(i%2 == 1 && j%2 == 0){
                        image2->verts[(y + (height - h - 1)) * width * cols + w + x] = image1->verts[(h * width + w)];
                    }
                    if(i%2 == 1 && j%2 == 1){
                        image2->verts[(y + (height - h - 1)) * width * cols + (width - w - 1) + x] = image1->verts[(h * width + w)];
                    }
                }
            }
        }
    }
    return image2;
}

struct ImageTile: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input<PrimitiveObject>("image");
        auto tilemode = get_input2<std::string>("tilemode");
        auto rows = get_input2<int>("rows");
        auto cols = get_input2<int>("cols");
        UserData &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        int w1 = w * cols;
        int h1 = h * rows;
        auto image2 = std::make_shared<PrimitiveObject>();
        image2->resize(w1 * h1);
        image2->userData().set2("isImage", 1);
        image2->userData().set2("w", w1);
        image2->userData().set2("h", h1);
        if(tilemode == "normal"){
            zeno::normal_tiling(image,image2, rows, cols);
        }
        if(tilemode == "mirror"){
            zeno::mirror_tiling(image,image2, rows, cols);
        }
        set_output("image", image2);
    }
};
ZENDEFNODE(ImageTile, {
    {
        {"image"},
        {"enum normal mirror", "tilemode", "normal"},
        {"int", "rows", "2"},
        {"int", "cols", "2"},
    },
    {
        {"image"},
    },
    {},
    {"comp"},
});

// ImageDilate函数实现
void imagedilate(std::shared_ptr<PrimitiveObject>& image, std::vector<std::vector<int>>& kernel,int iterations) {
    // 获取图像和卷积核的形状
    int image_height = image->userData().get2<int>("h");
    int image_width = image->userData().get2<int>("w");
    int kernel_height = kernel.size();
    int kernel_width = kernel[0].size();

    // 计算卷积核的中心点
    int center_y = kernel_height / 2;
    int center_x = kernel_width / 2;

    // 对每个像素进行膨胀操作,迭代
    for (int iter = 0; iter < iterations; iter++){
        auto imagetmp = std::make_shared<PrimitiveObject>();
        imagetmp->resize(image_width * image_height);
        imagetmp->userData().set2("isImage", 1);
        imagetmp->userData().set2("w", image_width);
        imagetmp->userData().set2("h", image_height);
        for (int y = center_y; y < image_height - center_y; y++) {
            for (int x = center_x; x < image_width - center_x; x++) {
                float maxValue0 = 0;
                float maxValue1 = 0;
                float maxValue2 = 0;
                // 遍历卷积核中的像素
                for (int ky = 0; ky < kernel_height; ky++) {
                    for (int kx = 0; kx < kernel_width; kx++) {
                        // 计算卷积核中的像素在原始图像中的位置
                        int image_y = y - center_y + ky;
                        int image_x = x - center_x + kx;
                        // 如果该位置是前景像素，则更新最大值
                        if (kernel[ky][kx] == 1 && image->verts[image_y * image_width + image_x][0] > maxValue0) {
                            maxValue0 = image->verts[image_y * image_width + image_x][0];
                        }
                        if (kernel[ky][kx] == 1 && image->verts[image_y * image_width + image_x][1] > maxValue1) {
                            maxValue1 = image->verts[image_y * image_width + image_x][1];
                        }
                        if (kernel[ky][kx] == 1 && image->verts[image_y * image_width + image_x][2] > maxValue2) {
                            maxValue2 = image->verts[image_y * image_width + image_x][2];
                        }
                    }
                }
                // 将最大值赋值给输出图像
                imagetmp->verts[y * image_width + x]= {maxValue0,maxValue1,maxValue2};
            }
        }
        image = imagetmp;
    }
}

struct ImageDilateED: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input<PrimitiveObject>("image");
        int strength = get_input2<int>("strength");
        int rows = get_input2<int>("kernel_rows");
        int cols = get_input2<int>("kernel_cols");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        std::vector<std::vector<int>> kernelvec(rows,std::vector<int>(cols,1));
        imagedilate(image,kernelvec,strength);
        set_output("image", image);
    }
};
ZENDEFNODE(ImageDilateED, {
    {
        {"image"},
        {"int", "strength", "1"},
        {"int", "kernel_rows", "3"},
        {"int", "kernel_cols", "3"},
    },
    {
        {"image"},
    },
    {},
    {"deprecated"},
});

// 图像膨胀函数
void dilateImage(cv::Mat& src, cv::Mat& dst, int kheight, int kwidth, int Strength) {
    // 定义结构元素
    cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(kheight, kwidth));
    // 进行膨胀操作
    cv::dilate(src, dst, kernel, cv::Point(-1, -1), Strength);
}
struct ImageDilate: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input<PrimitiveObject>("image");
        int strength = get_input2<int>("strength");
        int kheight = get_input2<int>("kernel_height");
        int kwidth = get_input2<int>("kernel_width");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        cv::Mat imagecvin(h, w, CV_32FC3);
        cv::Mat imagecvout(h, w, CV_32FC3);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                vec3f rgb = image->verts[i * w + j];
                imagecvin.at<cv::Vec3f>(i, j) = {rgb[0], rgb[1], rgb[2]};
            }
        }
        const int kernelSize = 3;
        dilateImage(imagecvin, imagecvout, kheight, kwidth, strength);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
                image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
            }
        }
        set_output("image", image);
    }
};
ZENDEFNODE(ImageDilate, {
    {
        {"image"},
        {"float", "strength", "1"},
        {"int", "kernel_width", "3"},
        {"int", "kernel_height", "3"},
    },
    {
        {"image"},
    },
    {},
    {"comp"},
});

void imageerode(std::shared_ptr<PrimitiveObject>& image, std::vector<std::vector<int>>& kernel, int iterations) {
    int image_height = image->userData().get2<int>("h");
    int image_width = image->userData().get2<int>("w");
    int kernel_height = kernel.size();
    int kernel_width = kernel[0].size();

    for (int iter = 0; iter < iterations; iter++) {
        auto imagetmp = std::make_shared<PrimitiveObject>();
        imagetmp->resize(image_width * image_height);
        imagetmp->userData().set2("isImage", 1);
        imagetmp->userData().set2("w", image_width);
        imagetmp->userData().set2("h", image_height);

        for (int i = 0; i < image_height; i++) {
            for (int j = 0; j < image_width; j++) {
                float minVal0 = 1;
                float minVal1 = 1;
                float minVal2 = 1;
                for (int x = 0; x < kernel_width; x++) {
                    for (int y = 0; y < kernel_height; y++) {
                        int posX = j + x - kernel_width / 2;
                        int posY = i + y - kernel_height / 2;
                        if (posX >= 0 && posX < image_width && posY >= 0 && posY < image_height) {
                            if (kernel[x][y] == 1) {
                                minVal0 = std::min(minVal0, image->verts[posY * image_width + posX][0]);
                                minVal1 = std::min(minVal1, image->verts[posY * image_width + posX][1]);
                                minVal2 = std::min(minVal2, image->verts[posY * image_width + posX][2]);
                            }
                        }
                    }
                }
                imagetmp->verts[i * image_width + j]= {minVal0,minVal1,minVal2};
            }
        }
        image = imagetmp;
    }
}

struct ImageErodeED: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input<PrimitiveObject>("image");
        int strength = get_input2<int>("strength");
        int rows = get_input2<int>("kernel_rows");
        int cols = get_input2<int>("kernel_cols");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        std::vector<std::vector<int>> kernelvec(rows,std::vector<int>(cols,1));
        imageerode(image, kernelvec, strength);
        set_output("image", image);
    }
};
ZENDEFNODE(ImageErodeED, {
    {
        {"image"},
        {"int", "strength", "1"},
        {"int", "kernel_rows", "3"},
        {"int", "kernel_cols", "3"},
    },
    {
        {"image"},
    },
    {},
    {"deprecated"},
});

struct ImageErode: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input<PrimitiveObject>("image");
        int strength = get_input2<int>("strength");
        int kheight = get_input2<int>("kernel_height");
        int kwidth = get_input2<int>("kernel_width");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        cv::Mat imagecvin(h, w, CV_32FC3);
        cv::Mat imagecvout(h, w, CV_32FC3);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                vec3f rgb = image->verts[i * w + j];
                imagecvin.at<cv::Vec3f>(i, j) = {rgb[0], rgb[1], rgb[2]};
            }
        }
        cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * kheight + 1, 2 * kwidth + 1),
                                                cv::Point(1, 1));
        cv::erode(imagecvin, imagecvout, kernel,cv::Point(-1, -1), strength);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
                image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
            }
        }
        set_output("image", image);
    }
};
ZENDEFNODE(ImageErode, {
    {
        {"image"},
        {"int", "strength", "1"},
        {"int", "kernel_width", "3"},
        {"int", "kernel_height", "3"},

    },
    {
        {"image"},
    },
    {},
    {"comp"},
});

struct ImageDilateByColor: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input<PrimitiveObject>("image");
        auto Hue = get_input2<std::string>("Hue");
        float Hi = get_input2<float>("H");
        float H = 0, S = 0, V = 0;
        const int kernelSize = 3;
        int kernel[kernelSize][kernelSize] = {
                {1, 1, 1},
                {1, 1, 1},
                {1, 1, 1}
        };
        if(Hue == "edit"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = Hi;

                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "red"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = 0;

                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "orange"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = 30;

                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "yellow"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = 60;

                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "green"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = 120;

                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "cyan"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = 180;

                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "blue"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = 240;

                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        if(Hue == "purple"){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                zeno::RGBtoHSV(R, G, B, H, S, V);
                H = 300;

                zeno::HSVtoRGB(H, S, V, R, G, B);
                image->verts[i][0] = R;
                image->verts[i][1] = G;
                image->verts[i][2] = B;
            }
        }
        set_output("image", image);
    }
};
ZENDEFNODE(ImageDilateByColor, {
    {
        {"image"},
        {"enum default edit red orange yellow green cyan blue purple ", "Hue", "edit"},
        {"float", "Hue", "0"},
        {"float", "distance", "30"},
    },
    {
        {"image"},
    },
    {},
    {"deprecated"},
});







}
}

