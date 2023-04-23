#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <zeno/zeno.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/NumericObject.h>

#include "tinyexr.h"
#include "zeno/utils/string.h"
#include <zeno/utils/scope_exit.h>
#include <stdexcept>


namespace zeno {

namespace {

//Test：https://blog.csdn.net/Angelloveyatou/article/details/130190085
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

void RGBtoHSV(float r, float g, float b, float &h, float &s, float &v) {
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

void HSVtoRGB(float h, float s, float v, float &r, float &g, float &b)
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
                H = H + (H - 0.5)*(Hi-1);
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

struct ImageEditBlur : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        auto xsize = get_input2<int>("xsize");
        auto ysize = get_input2<int>("ysize");
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
        if(mode=="Blur"){
            cv::blur(imagecvin,imagecvout,cv::Size(xsize,ysize),cv::Point(-1,-1));
        }
        if(mode=="GaussianBlur"){
            if(xsize%2==0){
                xsize += 1;
            }
            cv::GaussianBlur(imagecvin,imagecvout,cv::Size(xsize,xsize),1.5);
        }
        if(mode=="MedianBlur"){
            if(xsize%2==0){
                xsize += 1;
            }
            cv::medianBlur(imagecvin,imagecvout,xsize);
        }
        if(mode=="BilateralFilter"){
            if(xsize%2==0){
                xsize += 1;
            }
            cv::bilateralFilter(imagecvin,imagecvout,xsize,75,75);
        }
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
                image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageEditBlur, {
    {
        {"image"},
        {"enum Blur GaussianBlur MedianBlur BilateralFilter", "mode", "mode"},
        {"float", "xsize", "1"},
        {"float", "ysize", "1"},
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
        auto image = get_input<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        auto low_threshold = get_input2<float>("low_threshold");
        auto high_threshold = get_input2<float>("high_threshold");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

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
        }
        //TODO：error
        if(mode=="sobel"){
            cv::Mat imagecvin(h, w, CV_8U);
            cv::Mat imagecvout(h, w, CV_8U);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    imagecvin.at<uchar>(i, j) = int(rgb[0] * 255);
                }
            }
            cv::Mat gray(h, w, CV_8U);
            cv::Mat dx(h, w, CV_8U), dy(h, w, CV_8U);
            cv::cvtColor(imagecvin, gray, cv::COLOR_BGR2GRAY);
            cv::Sobel(gray,dx, CV_8U, 1, 0);
            cv::Sobel(gray,dy, CV_8U, 0, 1);
            cv::Mat angle(h, w, CV_8U);
            cv::cartToPolar(dx, dy, imagecvout, angle);
            cv::normalize(imagecvout, imagecvout, 0, 255, cv::NORM_MINMAX);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<uchar>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
        }
        if(mode=="scharr"){

        }
        if(mode=="laplacian"){

        }

        set_output("image", image);
    }
};

ZENDEFNODE(EdgeDetect, {
   {
        {"image"},
       {"enum canny sobel scharr laplacian", "mode", "canny"},
       {"float", "low_threshold", "100"},
       {"float", "high_threshold", "200"},
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
ZENDEFNODE(CompExtractRGBA, {
    {
        {"image"},
        {"enum RGBA R G B A", "RGBA", "RGBA"},
    },
    {
        {"image"}
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

        image->userData().set2("isImage", 1);
        image->userData().set2("w", nx);
        image->userData().set2("h", ny);

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
        auto x = get_input2<int>("x");
        auto y = get_input2<int>("y");

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
        {"int", "x", "256"},
        {"int", "y", "256"},
    },
    {
        {"image"}
        },
    {},
    { "comp" },
});

struct comp_color_ramp : INode {
    virtual void apply() override {

    }
};

ZENDEFNODE(comp_color_ramp, {
    {
        {"image"}
    },
    {
        {"image"}
        },
    {},
    { "" },
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

    }
};
ZENDEFNODE(CompNormalMap, {
    {
        {"image"}
    },
    {
        {"image"}
        },
    {},
    { "" },
});

struct ImageEditGray : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        for (auto i = 0; i < image->verts.size(); i++) {
            vec3f rgb = image->verts[i];
            float avg = (rgb[0] + rgb[1] + rgb[2]) / 3;
            image->verts[i] = {avg, avg, avg};
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageEditGray, {
    {
        {"image"}
    },
    {
        {"image"}
        },
    {},
    { "comp" },
});

/* 调整黑点、白点和中值以增加、平衡或降低对比度。
您可以使用Value选项卡全局调整级别(影响所有通道)，或使用R、
G、B或Comp 4选项卡逐个通道调整。输入级别用于压缩黑点和白点
范围，从而增加对比度。
Gamma将作为使用输入级别指定的范围的中值调整进行应用。输出级
别扩展了黑白点范围，降低了对比度。 */

struct CompLevels : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        for (auto i = 0; i < image->verts.size(); i++) {

        }
        set_output("image", image);
    }
};

ZENDEFNODE(CompLevels, {
    {
        {"image"}
    },
    {
        {"image"}
        },
    {},
    { "" },
});

/* 此操作将输入数据量化为离散的步骤，从而降低图像中的颜色级别。 */
struct comp_quantize : INode {
    virtual void apply() override {

    }
};

ZENDEFNODE(comp_quantize, {
    {
    },
    {},
    {},
    { "" },
});
}
}
