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

struct ImageEdit_RGB : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto RGB = get_input2<std::string>("RGB");
        auto Average = get_input2<bool>("Average");
        auto Invert = get_input2<bool>("Invert");

        float R = get_input2<float>("R");
        float G = get_input2<float>("G");
        float B = get_input2<float>("B");
        float L = get_input2<float>("Luminace");
        float ContrastRatio = get_input2<float>("ContrastRatio");

        if(RGB == "RGB") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = R * image->verts[i][0];
                float G1 = G * image->verts[i][1];
                float B1 = B * image->verts[i][2];
                image->verts[i][0] = R1 * L;
                image->verts[i][1] = G1 * L;
                image->verts[i][2] = B1 * L;
            }
        }
        if(RGB == "R") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = R * image->verts[i][0];
                float G1 = 0;
                float B1 = 0;
                image->verts[i][0] = R1 * L;
                image->verts[i][1] = G1 * L;
                image->verts[i][2] = B1 * L;
            }
        }
        if(RGB == "G") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = 0;
                float G1 = G * image->verts[i][1];
                float B1 = 0;
                image->verts[i][0] = R1 * L;
                image->verts[i][1] = G1 * L;
                image->verts[i][2] = B1 * L;
            }
        }
        if(RGB == "B") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = 0;
                float G1 = 0;
                float B1 = B * image->verts[i][2];
                image->verts[i][0] = R1 * L;
                image->verts[i][1] = G1 * L;
                image->verts[i][2] = B1 * L;
            }
        }
        for (auto i = 0; i < image->verts.size(); i++) {
            image->verts[i] = image->verts[i] + (image->verts[i]-0.5) * (ContrastRatio-1);
        }

        if(Average){
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

ZENDEFNODE(ImageEdit_RGB, {
    {
        {"image"},
        {"enum RGB R G B", "RGB", "RGB"},
        {"float", "R", "1"},
        {"float", "G", "1"},
        {"float", "B", "1"},
        {"float", "Luminace", "1"},
        {"float", "ContrastRatio", "1"},
        {"bool", "Average", "0"},
        {"bool", "Invert", "0"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

/* 图像模糊，可以使用不同的卷积核(Gaussian) */
struct CompBlur : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto blur = get_input2<float>("blur");
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
        cv::blur(imagecvin,imagecvout,cv::Size(blur,blur),cv::Point(-1,-1));
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
                image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(CompBlur, {
    {
        {"image"},
        {"float", "blur", "16"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

/*边缘查找*/
struct CompEdgeDetect : INode {
    virtual void apply() override {

    }
};

ZENDEFNODE(CompEdgeDetect, {
    {
        {"image"}
    },
    {},
    {},
    { "" },
});

//边缘检测
struct CVCanny : INode {
    void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto low_threshold = get_input2<float>("low_threshold");
        auto high_threshold = get_input2<float>("high_threshold");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
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
};

ZENDEFNODE(CVCanny, {
   {
        {"image"},
       {"float", "low_threshold", "100"},
       {"float", "high_threshold", "200"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});


/* 合成图层。可能需要的参数：前景图层，前景权重，背景图层，背景
权重，Mask图层，Mask权重
用 mask 将前景和背景叠加的操作算法（操作分别作用在前背景和alpha通道上）：
mask可以被忽略，它只是将操作限制在图像的一个区域。mask可
以是反转的、变亮的或变暗的。
mask可以单独指定，或者来自前景的alpha通道。
算法参考：https://www.deanhan.cn/canvas-blende-mode.html
over：将前景放置于背景之上
under：将前景放置在背景的alpha之下
atop：当背景的alpha存在的时候，才将前景放置于背景之上
inside：将前景放置在背景的alpha中。
outside：将前景放置在背景的alpha之外
screen：作用与饱和度add相同
add：将前景添加到背景上
subtract：从背景中减去前景
diff：获取前景和背景之间的差的绝对值
myltiply：将背景与前景相乘
minimum：取前景和背景的最小值
maximum：取前景和背景的最大值
average：取前景和背景的平均值
xor：异或运算，对两个Alpha平面进行异或运算，以便删除重叠的
Alpha区域
图像过滤器（可能在分辨率不统一的时候使用） */
struct Composite3: INode {
    virtual void apply() override {
        auto image1 = get_input<PrimitiveObject>("Foreground");
        auto image2 = get_input<PrimitiveObject>("Background");
        auto image3 = get_input<PrimitiveObject>("Mask");
        auto compmode = get_input2<std::string>("compmode");
        auto &ud1 = image1->userData();
        int w1 = ud1.get2<int>("w");
        int h1 = ud1.get2<int>("h");

        if(compmode == "Add"){
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    float r1 = image1->verts[i * w1 + j][0];
                    float r2 = image2->verts[i * w1 + j][0];
                    float g1 = image1->verts[i * w1 + j][1];
                    float g2 = image2->verts[i * w1 + j][1];
                    float b1 = image1->verts[i * w1 + j][2];
                    float b2 = image2->verts[i * w1 + j][2];
                    float l = image3->verts[i * w1 + j][0];
                    image1->verts[i * w1 + j] = {r1*l+r2*(1-l), g1*l+g2*(1-l), b1*l+b2*(1-l)};
                }
            }
        }
        if(compmode == "Subtract"){
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    float r1 = image1->verts[i * w1 + j][0];
                    float r2 = image2->verts[i * w1 + j][0];
                    float g1 = image1->verts[i * w1 + j][1];
                    float g2 = image2->verts[i * w1 + j][1];
                    float b1 = image1->verts[i * w1 + j][2];
                    float b2 = image2->verts[i * w1 + j][2];
                    float l = image3->verts[i * w1 + j][0];
                    image1->verts[i * w1 + j] = {r1*l-r2*(1-l), g1*l-g2*(1-l), b1*l-b2*(1-l)};
                }
            }
        }
        if(compmode == "Multiply"){
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    float r1 = image1->verts[i * w1 + j][0];
                    float r2 = image2->verts[i * w1 + j][0];
                    float g1 = image1->verts[i * w1 + j][1];
                    float g2 = image2->verts[i * w1 + j][1];
                    float b1 = image1->verts[i * w1 + j][2];
                    float b2 = image2->verts[i * w1 + j][2];
                    float l = image3->verts[i * w1 + j][0];
                    image1->verts[i * w1 + j] = {r1*l*r2*(1-l), g1*l*g2*(1-l), b1*l*b2*(1-l)};
                }
            }
        }
        if(compmode == "Diff"){
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    float r1 = image1->verts[i * w1 + j][0];
                    float r2 = image2->verts[i * w1 + j][0];
                    float g1 = image1->verts[i * w1 + j][1];
                    float g2 = image2->verts[i * w1 + j][1];
                    float b1 = image1->verts[i * w1 + j][2];
                    float b2 = image2->verts[i * w1 + j][2];
                    float l = image3->verts[i * w1 + j][0];
                    image1->verts[i * w1 + j] = {r1*l/(r2*(1-l)), g1*l/(g2*(1-l)), b1*l/(b2*(1-l))};
                }
            }
        }
        set_output("image", image1);
    }
};

ZENDEFNODE(Composite3, {
    {
        {"Foreground"},
        {"Background"},
        {"Mask"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

struct Composite2: INode {
    virtual void apply() override {
        auto image1 = get_input<PrimitiveObject>("Foreground");
        auto image2 = get_input<PrimitiveObject>("Background");
        auto compmode = get_input2<std::string>("compmode");
        auto &ud1 = image1->userData();
        int w1 = ud1.get2<int>("w");
        int h1 = ud1.get2<int>("h");
        auto &ud2 = image2->userData();
        int w2 = ud2.get2<int>("w");
        int h2 = ud2.get2<int>("h");
        if(compmode == "Over"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha = image1->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l = alpha[i * w1 + j];
                        image1->verts[i * w1 + j] = {r1 * l + r2 * (1 - l), g1 * l + g2 * (1 - l),
                                                     b1 * l + b2 * (1 - l)};
                    }
                }
            }
        }
        if(compmode == "Under"){
            if (image2->verts.has_attr("alpha")) {
                auto &alpha = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l = alpha[i * w1 + j];
                        image1->verts[i * w1 + j] = {r2 * l + r1 * (1 - l), g2 * l + g1 * (1 - l),
                                                     b2 * l + b1 * (1 - l)};
                    }
                }
            }
        }
        if(compmode == "AtopB"){
            if (image2->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        if(l2==1){
                            image1->verts[i * w1 + j] = {r1 * l1 + r2 * l2, g1 * l1 + g2 * l2,
                                                         b1 * l1 + b2 * l2};
                        }
                        else{
                            image1->verts[i * w1 + j] = {0,0,0};
                        }
                    }
                }
            }
        }
        if(compmode == "BtopA"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        if(l1==1){
                            image1->verts[i * w1 + j] = {r1 * l1 + r2 * l2, g1 * l1 + g2 * l2,
                                                         b1 * l1 + b2 * l2};
                        }
                        else{
                            image1->verts[i * w1 + j] = {0,0,0};
                        }
                    }
                }
            }
        }
        if(compmode == "Inside"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        if(l1==1 && l2==1){
                            image1->verts[i * w1 + j] = {r1 * l1, g1 * l1,
                                                         b1 * l1 };
                        }
                        else{
                            image1->verts[i * w1 + j] = {0,0,0};
                        }
                    }
                }
            }
        }
        if(compmode == "Outside"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        if(l1==1 && l2==0){
                            image1->verts[i * w1 + j] = {r1 * l1, g1 * l1,
                                                         b1 * l1 };
                        }
                        else{
                            image1->verts[i * w1 + j] = {0,0,0};
                        }
                    }
                }
            }
        }
        if(compmode == "Screen"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        if(l1==1 && l2==0){
                            image1->verts[i * w1 + j] = {r1 * l1, g1 * l1,
                                                         b1 * l1 };
                        }
                        else{
                            image1->verts[i * w1 + j] = {0,0,0};
                        }
                    }
                }
            }
        }
        if(compmode == "Add"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        image1->verts[i * w1 + j] = {r1 * l1 + r2 * l2, g1 * l1 + g2 * l2,
                                                     b1 * l1 + b2 * l2};
                    }
                }
            }
        }
        if(compmode == "Subtract"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        image1->verts[i * w1 + j] = {r1 * l1 - r2 * l2, g1 * l1 - g2 * l2,
                                                     b1 * l1 - b2 * l2};
                    }
                }
            }
        }
        if(compmode == "Multiply"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        image1->verts[i * w1 + j] = {r1 * l1 * r2 * l2, g1 * l1 * g2 * l2,
                                                     b1 * l1 * b2 * l2};
                    }
                }
            }
        }
        if(compmode == "Divide"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        image1->verts[i * w1 + j] = {r1 * l1 / (r2 * l2), g1 * l1 / (g2 * l2),
                                                     b1 * l1 / (b2 * l2)};
                    }
                }
            }
        }
        if(compmode == "Diff"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        image1->verts[i * w1 + j] = {abs(r1 * l1 - r2 * l2), abs(g1 * l1 - g2 * l2),
                                                     abs(b1 * l1 - b2 * l2)};
                    }
                }
            }
        }
        if(compmode == "Minimum"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        image1->verts[i * w1 + j] = {std::min(r1 * l1 , r2 * l2), std::min(g1 * l1 , g2 * l2),
                                                     std::min(b1 * l1 , b2 * l2)};
                    }
                }
            }
        }
        if(compmode == "Maxinum"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        image1->verts[i * w1 + j] = {std::max(r1 * l1 , r2 * l2), std::max(g1 * l1 , g2 * l2),
                                                     std::max(b1 * l1 , b2 * l2)};
                    }
                }
            }
        }
        if(compmode == "Average"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        image1->verts[i * w1 + j] = {(r1 * l1 + r2 * l2)/2,(g1 * l1 + g2 * l2)/2,
                                                    (b1 * l1 + b2 * l2)/2};
                    }
                }
            }
        }
        if(compmode == "Xor"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        if(!(l1==1 && l2==1)){
                            image1->verts[i * w1 + j] = {r1 * l1 + r2 * l2, g1 * l1 + g2 * l2,
                                                         b1 * l1 + b2 * l2};
                        }
                        else{
                            image1->verts[i * w1 + j] = {0,0,0};
                        }
                    }
                }
            }
        }
        if(compmode == "Alpha"){
            if (image1->verts.has_attr("alpha")) {
                auto &alpha1 = image1->verts.attr<float>("alpha");
                auto &alpha2 = image2->verts.attr<float>("alpha");
                for (int i = 0; i < h1; i++) {
                    for (int j = 0; j < w1; j++) {
                        float r1 = image1->verts[i * w1 + j][0];
                        float r2 = image2->verts[i * w1 + j][0];
                        float g1 = image1->verts[i * w1 + j][1];
                        float g2 = image2->verts[i * w1 + j][1];
                        float b1 = image1->verts[i * w1 + j][2];
                        float b2 = image2->verts[i * w1 + j][2];
                        float l1 = alpha1[i * w1 + j];
                        float l2 = alpha2[i * w1 + j];
                        if(l1==1 || l2==1){
                            image1->verts[i * w1 + j] = {1,1,1};
                        }
                        else{
                            image1->verts[i * w1 + j] = {0,0,0};
                        }
                    }
                }
            }
        }
        set_output("image", image1);
    }
};

ZENDEFNODE(Composite2, {
    {
        {"Foreground"},
        {"Background"},
        {"enum Over Under AtopB BtopA Inside Outside Screen Add Subtract Multiply Divide Diff Minimum Maxinum Average Xor Alpha ", "compmode", "Over"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

struct Composite1: INode {
    virtual void apply() override {
        auto image1 = get_input<PrimitiveObject>("Foreground");
        auto image2 = get_input<PrimitiveObject>("Background");
        auto mode = get_input2<std::string>("mode");
        float ratio = get_input2<float>("ratio");
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
            cv::addWeighted(imagecvin1, 1 - ratio, imagecvin2, ratio, 0, imagecvout);
        }
        if(mode == "Subtract"){
            cv::subtract(imagecvin1*ratio, imagecvin2*(1-ratio), imagecvout);
        }
        if(mode == "Multiply"){
            cv::multiply(imagecvin1*ratio, imagecvin2*(1-ratio), imagecvout);
        }
        if(mode == "Divide"){
            cv::absdiff(imagecvin1*ratio, imagecvin2*(1-ratio), imagecvout);
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

ZENDEFNODE(Composite1, {
    {
        {"Foreground"},
        {"Background"},
        {"enum Add Subtract Multiply Divide", "mode", "Add"},
        {"float", "ratio", "0.5"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

//RGB2YUV BT.709标准
struct ImageEdit_YUV : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto RGB = get_input2<std::string>("RGB");
        auto Gray = get_input2<bool>("Gray_BT.709");
        auto Average = get_input2<bool>("Average");
        auto Invert = get_input2<bool>("Invert");

        float R = get_input2<float>("R");
        float G = get_input2<float>("G");
        float B = get_input2<float>("B");
        float L = get_input2<float>("Luminace_BT.709");
        float ContrastRatio = get_input2<float>("ContrastRatio");

        if(RGB == "RGB") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = R * image->verts[i][0];
                float G1 = G * image->verts[i][1];
                float B1 = B * image->verts[i][2];
                float Y = L * (0.2126 * R1 + 0.7152 * G1 + 0.0722 * B1);
                float U = -0.1145 * R1 - 0.3855 * G1 + 0.500 * B1;
                float V = 0.500 * R1 - 0.4543 * G1 - 0.0457 * B1;
                image->verts[i][0] = Y + 1.5748 * V;
                image->verts[i][1] = Y - 0.1868 * U - 0.4680 * V;
                image->verts[i][2] = Y + 1.856 * U;
            }
        }
        if(RGB == "R") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = R * image->verts[i][0];
                float G1 = 0;
                float B1 = 0;
                float Y = L * (0.2126 * R1 + 0.7152 * G1 + 0.0722 * B1);
                float U = -0.1145 * R1 - 0.3855 * G1 + 0.500 * B1;
                float V = 0.500 * R1 - 0.4543 * G1 - 0.0457 * B1;
                image->verts[i][0] = Y + 1.5748 * V;
                image->verts[i][1] = Y - 0.1868 * U - 0.4680 * V;
                image->verts[i][2] = Y + 1.856 * U;
            }
        }
        if(RGB == "G") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = 0;
                float G1 = G * image->verts[i][1];
                float B1 = 0;
                float Y = L * (0.2126 * R1 + 0.7152 * G1 + 0.0722 * B1);
                float U = -0.1145 * R1 - 0.3855 * G1 + 0.500 * B1;
                float V = 0.500 * R1 - 0.4543 * G1 - 0.0457 * B1;
                image->verts[i][0] = Y + 1.5748 * V;
                image->verts[i][1] = Y - 0.1868 * U - 0.4680 * V;
                image->verts[i][2] = Y + 1.856 * U;
            }
        }
        if(RGB == "B") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = 0;
                float G1 = 0;
                float B1 = B * image->verts[i][2];
                float Y = L * (0.2126 * R1 + 0.7152 * G1 + 0.0722 * B1);
                float U = -0.1145 * R1 - 0.3855 * G1 + 0.500 * B1;
                float V = 0.500 * R1 - 0.4543 * G1 - 0.0457 * B1;
                image->verts[i][0] = Y + 1.5748 * V;
                image->verts[i][1] = Y - 0.1868 * U - 0.4680 * V;
                image->verts[i][2] = Y + 1.856 * U;
            }
        }
        for (auto i = 0; i < image->verts.size(); i++) {
            image->verts[i] =  image->verts[i]+(image->verts[i]-0.5) * ContrastRatio;
        }
        if(Gray){
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                float G = image->verts[i][1];
                float B = image->verts[i][2];
                image->verts[i][0] = 0.2126 * R + 0.7152 * G + 0.0722 * B ;
                image->verts[i][1] = 0.2126 * R + 0.7152 * G + 0.0722 * B ;
                image->verts[i][2] = 0.2126 * R + 0.7152 * G + 0.0722 * B ;
            }
        }
        if(Average){
            for (auto i = 0; i < image->verts.size(); i++) {
                float avr = (R + G + B)/3;
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


ZENDEFNODE(ImageEdit_YUV, {
    {
        {"image"},
        {"enum RGB R G B", "RGB", "RGB"},
        {"float", "R", "1"},
        {"float", "G", "1"},
        {"float", "B", "1"},
        {"float", "Luminace_BT.709", "1"},
        {"float", "ContrastRatio", "1"},
        {"bool", "Gray_BT.709", "0"},
        {"bool", "Average", "0"},
        {"bool", "Invert", "0"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

struct ImageEdit_HSV : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        set_output("image", image);
    }
};

ZENDEFNODE(ImageEdit_HSV, {
    {
        {"image"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

/* 提取RGB */
struct ExtractRGBA : INode {
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
ZENDEFNODE(ExtractRGBA, {
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

/* 删除指定的图层(属性)。需要指定图层的名称（可能会有多个），选
项：删除选择/未选择图层 */
struct CompDelete : INode {
    virtual void apply() override {

    }
};

ZENDEFNODE(CompDelete, {
    {
    },
    {
        {"image"}
        },
    {},
    { "" },
});

/* 重命名图层，可能需要的参数：源名称，目标名称 */
struct CompRename : INode {
    virtual void apply() override {

    }
};

ZENDEFNODE(CompRename, {
    {
    },
    {
        {"image"}
        },
    {},
    { "" },
});

/* 创建颜色图层，可能需要的参数：颜色，分辨率，图层名称 */
struct CompColor : INode {
    virtual void apply() override {

    }
};

ZENDEFNODE(CompColor, {
    {
        {"image"}
    },
    {
        {"image"}
        },
    {},
    { "" },
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


/* 图像对比度调节
此操作可增加或降低图像的对比度。这可以通过两种方式实现：
范围-通过设置原始黑白的新值。该范围将被重新映射以适应新值。
缩放-通过拾取中心轴(通常为0.5)并围绕该值进行缩放。 */
struct CompContrast : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        float ContrastRatio = get_input2<float>("ContrastRatio");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        for (auto i = 0; i < image->verts.size(); i++) {
            image->verts[i] =  image->verts[i]+(image->verts[i]-0.5) * ContrastRatio;
        }
        set_output("image", image);
    }
};

ZENDEFNODE(CompContrast, {
    {
        {"image"},
        {"float", "ContrastRatio", "1"},
    },
    {"image"},
    {},
    { "" },
});

/* 图像饱和度调节() */
struct CompSaturation : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        float Si = get_input2<float>("Saturation");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        for (auto i = 0; i < image->verts.size(); i++) {

            float R = image->verts[i][0];
            float G = image->verts[i][1];
            float B = image->verts[i][2];
//            float S = Si * (1-(3 * zeno::min(zeno::min(R,G),B))/(R+G+B));
//            float V = (R+G+B)/3;
//            float m = V -
            image->verts[i][0] = R * Si;
            image->verts[i][1] = G * Si;
            image->verts[i][2] = B * Si;
        }
        set_output("image", image);
    }
};

ZENDEFNODE(CompSaturation, {
    {
        {"image"},
        {"float", "ContrastRatio", "1"},
    },
    {"image"},
    {},
    { "" },
});


/* 对图像应用像素反转，本质上是 Clr_out = 1 - Clr_in */
struct CompInvert : INode{
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
ZENDEFNODE(CompInvert, {
    {
        {"image"},
    },
    {
        "image",
    },
    {},
    {""},
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

/* 此操作将颜色或向量转换为标量，如亮度或长度。或者，可以将向量
平面转换为标量平面。 */
struct CompAverage : INode {
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

ZENDEFNODE(CompAverage, {
    {
        {"image"}
    },
    {
        {"image"}
        },
    {},
    { "" },
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
