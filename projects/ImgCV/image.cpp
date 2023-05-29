#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/scope_exit.h>
#include <stdexcept>
#include <cmath>
#include <zeno/utils/log.h>
#include <opencv2/videoio.hpp>
#include <filesystem>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

//class SIFT;

using namespace cv;
//using namespace cv::xfeatures2d;

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

struct ImageResize: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input<PrimitiveObject>("image");
        int width = get_input2<int>("width");
        int height = get_input2<int>("height");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        auto image2 = std::make_shared<PrimitiveObject>();
        image2->verts.resize(width * height);
        image2->userData().set2("isImage", 1);
        image2->userData().set2("w", width);
        image2->userData().set2("h", height);
        if(image->has_attr("alpha")){
            image2->verts.add_attr<float>("alpha");
        }
        // 计算尺寸比例
        float scaleX = static_cast<float>(w) / width;
        float scaleY = static_cast<float>(h) / height;
        // 改变图像大小
        for (auto a = 0; a < image->verts.size(); a++){
            int x = a / w;
            int y = a % w;
            int srcX = static_cast<int>(x * scaleX);
            int srcY = static_cast<int>(y * scaleY);
            image2->verts[y * width + x] = image->verts[srcY * w + srcX];
            image2->verts.attr<float>("alpha")[y * width + x] = image->verts.attr<float>("alpha")[srcY * w + srcX];
        }
        set_output("image", image2);
    }
};
ZENDEFNODE(ImageResize, {
    {
        {"image"},
        {"int", "width", "1024"},
        {"int", "height", "1024"},
    },
    {
        {"image"},
    },
    {},
    {"image"},
});

void rotateimage(std::shared_ptr<PrimitiveObject> src, std::shared_ptr<PrimitiveObject> & dst, float angle, bool balpha) {
    // 计算旋转角度的弧度值
    double radians = angle * 3.14159 / 180.0;
    int width = src->userData().get2<int>("w");
    int height = src->userData().get2<int>("h");

    // 计算旋转中心点
    int centerX = width / 2;
    int centerY = height / 2;

    // 计算旋转后的图像宽度和高度
    int rotatedWidth = static_cast<int>(std::abs(width * cos(radians)) + std::abs(height * sin(radians)));
    int rotatedHeight = static_cast<int>(std::abs(height * cos(radians)) + std::abs(width * sin(radians)));

    dst->verts.resize(rotatedWidth * rotatedHeight);
    dst->userData().set2("w", rotatedWidth);
    dst->userData().set2("h", rotatedHeight);
    if(src->verts.has_attr("alpha")){
        dst->verts.add_attr<float>("alpha");
        if(balpha){
            for(int i = 0;i<dst->size();i++){
                dst->verts.attr<float>("alpha")[i] = 0;
            }
        }
        else{
            for(int i = 0;i<dst->size();i++){
                dst->verts.attr<float>("alpha")[i] = 1;
            }
        }
        for (int y = 0; y < rotatedHeight; ++y) {
            for (int x = 0; x < rotatedWidth; ++x) {
                // 计算旋转前的坐标
                int srcX = static_cast<int>((x - rotatedWidth / 2) * cos(-radians) - (y - rotatedHeight / 2) * sin(-radians) + centerX);
                int srcY = static_cast<int>((x - rotatedWidth / 2) * sin(-radians) + (y - rotatedHeight / 2) * cos(-radians) + centerY);
                // 检查坐标是否在旋转图像范围内
                if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                    // 获取旋转前的像素值
                    dst->verts[y * rotatedWidth + x] = src->verts[srcY * width + srcX] ;
                    dst->verts.attr<float>("alpha")[y * rotatedWidth + x] = src->verts.attr<float>("alpha")[srcY * width + srcX];
                }
            }
        }
    }
    else{
        dst->verts.add_attr<float>("alpha");
        if(balpha){
            for(int i = 0;i<dst->size();i++){
                dst->verts.attr<float>("alpha")[i] = 0;
            }
        }
        else{
            for(int i = 0;i<dst->size();i++){
                dst->verts.attr<float>("alpha")[i] = 1;
            }
        }
        for (int y = 0; y < rotatedHeight; ++y) {
            for (int x = 0; x < rotatedWidth; ++x) {
                // 计算旋转前的坐标
                int srcX = static_cast<int>((x - rotatedWidth / 2) * cos(-radians) - (y - rotatedHeight / 2) * sin(-radians) + centerX);
                int srcY = static_cast<int>((x - rotatedWidth / 2) * sin(-radians) + (y - rotatedHeight / 2) * cos(-radians) + centerY);
                // 检查坐标是否在旋转图像范围内
                if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                    // 获取旋转前的像素值
                    dst->verts[y * rotatedWidth + x] = src->verts[srcY * width + srcX] ;
                    dst->verts.attr<float>("alpha")[y * rotatedWidth + x] = 1;
                }
            }
        }
    }
    // 遍历旋转后的图像像素
    for (int y = 0; y < rotatedHeight; ++y) {
        for (int x = 0; x < rotatedWidth; ++x) {
            // 计算旋转前的坐标
            int srcX = static_cast<int>((x - rotatedWidth / 2) * cos(-radians) - (y - rotatedHeight / 2) * sin(-radians) + centerX);
            int srcY = static_cast<int>((x - rotatedWidth / 2) * sin(-radians) + (y - rotatedHeight / 2) * cos(-radians) + centerY);

            // 检查坐标是否在旋转图像范围内
            if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                // 获取旋转前的像素值
                dst->verts[y * rotatedWidth + x] = src->verts[srcY * width + srcX] ;
            }
        }
    }
}
struct ImageRotate: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input<PrimitiveObject>("image");
        auto balpha = get_input2<bool>("alpha");
        float rotate = get_input2<float>("rotate");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        auto image2 = std::make_shared<PrimitiveObject>();
        image2->verts.resize(w * h);
        image2->userData().set2("isImage", 1);
        rotateimage(image,image2,rotate,balpha);
        auto &ud2 = image2->userData();
        w = ud2.get2<int>("w");
        h = ud2.get2<int>("h");
        set_output("image", image2);
    }
};
ZENDEFNODE(ImageRotate, {
    {
        {"image"},
        {"float", "rotate", "0.0"},
        {"bool", "alpha", "0"},
    },
    {
        {"image"},
    },
    {},
    {"image"},
});

std::shared_ptr<PrimitiveObject> flipimage(std::shared_ptr<PrimitiveObject> src, bool flipX, bool flipY) {
    auto &ud = src->userData();
    int w = ud.get2<int>("w");
    int h = ud.get2<int>("h");
    auto imagefxy = std::make_shared<PrimitiveObject>();
    imagefxy->verts.resize(w * h);
    imagefxy->userData().set2("isImage", 1);
    imagefxy->userData().set2("w", w);
    imagefxy->userData().set2("h", h);
    if(src->verts.has_attr("alpha")) {
        imagefxy->verts.add_attr<float>("alpha");
        imagefxy->verts.attr<float>("alpha") = src->verts.attr<float>("alpha");
    }
    if(flipX && flipY){
        if(imagefxy->verts.has_attr("alpha")){
            for(int i = 0;i < h;i++){
                for(int j = 0;j < w;j++){
                    imagefxy->verts[i * w + j] = src->verts[(h-i-1) * w + (w-j-1)];
                    imagefxy->verts.attr<float>("alpha")[i * w + j] = src->verts.attr<float>("alpha")[(h-i-1) * w + (w-j-1)];
                }
            }
        }
        for(int i = 0;i < h;i++){
            for(int j = 0;j < w;j++){
                imagefxy->verts[i * w + j] = src->verts[(h-i-1) * w + (w-j-1)];
            }
        }
    }
    else if(flipX && !flipY){
        if(imagefxy->verts.has_attr("alpha")){
            for(int i = 0;i < h;i++){
                for(int j = 0;j < w;j++){
                    imagefxy->verts[i * w + j] = src->verts[(h-i-1) * w + j];
                    imagefxy->verts.attr<float>("alpha")[i * w + j] = src->verts.attr<float>("alpha")[(h-i-1) * w + j];
                }
            }
        }
        for(int i = 0;i < h;i++){
            for(int j = 0;j < w;j++){
                imagefxy->verts[i * w + j] = src->verts[(h-i-1) * w + j];
            }
        }
    }
    else if(!flipX && flipY){
        if(imagefxy->verts.has_attr("alpha")){
            for(int i = 0;i < h;i++){
                for(int j = 0;j < w;j++){
                    imagefxy->verts[i * w + j] = src->verts[i * w + (w-j-1)];
                    imagefxy->verts.attr<float>("alpha")[i * w + j] = src->verts.attr<float>("alpha")[i * w + (w-j-1)];
                }
            }
        }
        for(int i = 0;i < h;i++){
            for(int j = 0;j < w;j++){
                imagefxy->verts[i * w + j] = src->verts[i * w + (w-j-1)];
            }
        }
    }
    else if(!flipX && !flipY){
        imagefxy->verts = src->verts;
    }
    return imagefxy;
}

struct ImageFlip: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto flipX = get_input2<bool>("flipX");
        auto flipY = get_input2<bool>("flipY");
        set_output("image", flipimage(image,flipX,flipY));
    }
};

ZENDEFNODE(ImageFlip, {
    {
        {"image"},
        {"bool", "flipX", "0"},
        {"bool", "flipY", "0"},
    },
    {
        {"image"},
    },
    {},
    {"image"},
});

struct Composite: INode {
    virtual void apply() override {
        auto compmode = get_input2<std::string>("Compmode");
        auto maskmode1 = get_input2<std::string>("Mask1mode");
        auto maskmode2 = get_input2<std::string>("Mask2mode");
        int w1 = 1024 ;
        int h1 = 1024 ;
        auto image1 = std::make_shared<PrimitiveObject>();
        image1->verts.resize(w1 * h1);
        image1->userData().set2("isImage", 1);
        image1->userData().set2("w", w1);
        image1->userData().set2("h", h1);
        auto image2 = std::make_shared<PrimitiveObject>();
        image2->verts.resize(w1 * h1);
        image2->userData().set2("isImage", 1);
        image2->userData().set2("w", w1);
        image2->userData().set2("h", h1);
        auto A1 = std::make_shared<PrimitiveObject>();
        A1->verts.resize(w1 * h1);
        A1->userData().set2("isImage", 1);
        A1->userData().set2("w", w1);
        A1->userData().set2("h", h1);
        A1->verts.add_attr<float>("alpha");
        for(int i = 0;i < w1 * h1;i++){
            A1->verts.attr<float>("alpha")[i] = 0.0;
        }
        std::vector<float> &alpha1 = A1->verts.attr<float>("alpha");
        auto A2 = std::make_shared<PrimitiveObject>();
        A2->verts.resize(w1 * h1);
        A2->userData().set2("isImage", 1);
        A2->userData().set2("w", w1);
        A2->userData().set2("h", h1);
        A2->verts.add_attr<float>("alpha");
        for(int i = 0;i < w1 * h1;i++){
            A2->verts.attr<float>("alpha")[i] = 0.0;
        }
        std::vector<float> &alpha2 = A2->verts.attr<float>("alpha");

        if(has_input("Background")){
            image2 = get_input2<PrimitiveObject>("Background");
            auto &ud2 = image2->userData();
            w1 = ud2.get2<int>("w");
            h1 = ud2.get2<int>("h");
            if(image2->verts.has_attr("alpha")){
                alpha2 = image2->verts.attr<float>("alpha");
            }
            else{
                image2->verts.add_attr<float>("alpha");
                for(int i = 0;i < w1 * h1;i++){
                    image2->verts.attr<float>("alpha")[i] = 1.0;
                }
                alpha2 = image2->verts.attr<float>("alpha");
            }
            if(!has_input("Foreground")){
                image1->verts.resize(image2->size());
                image1->userData().set2("w", w1);
                image1->userData().set2("h", h1);
                image1->verts.add_attr<float>("alpha");
                for(int i = 0;i < w1 * h1;i++){
                    image1->verts.attr<float>("alpha")[i] = 0.0;
                }
                alpha1 = image1->verts.attr<float>("alpha");
            }
        }
        if(has_input("Foreground")){
            image1 = get_input2<PrimitiveObject>("Foreground");
            auto &ud1 = image1->userData();
            w1 = ud1.get2<int>("w");
            h1 = ud1.get2<int>("h");
            if(image1->verts.has_attr("alpha")){
                alpha1 = image1->verts.attr<float>("alpha");
            }
            else{
                image1->verts.add_attr<float>("alpha");
                for(int i = 0;i < w1 * h1;i++){
                    image1->verts.attr<float>("alpha")[i] = 1.0;
                }
                alpha1 = image1->verts.attr<float>("alpha");
            }
            if(!has_input("Background")){
                image2->verts.resize(image1->size());
                image2->userData().set2("w", w1);
                image2->userData().set2("h", h1);
                image2->verts.add_attr<float>("alpha");
                for(int i = 0;i < w1 * h1;i++){
                    image2->verts.attr<float>("alpha")[i] = 0.0;
                }
                alpha2 = image2->verts.attr<float>("alpha");
            }
            if(has_input("Background")){
                auto &ud2 = image2->userData();
                int w2 = ud2.get2<int>("w");
                int h2 = ud2.get2<int>("h");
                if(image1->size() != image2->size() || w1 != w2 || h1 != h2){
                    image2->verts.resize(image1->size());
                    image2->userData().set2("w", w1);
                    image2->userData().set2("h", h1);
//todo： image1和image2大小不同的情况
//                    for (int i = 0; i < h1; i++) {
//                        for (int j = 0; j < w1; j++) {
//
//                        }
//                    }
                }
            }
        }
        if(has_input("Mask1")) {
            auto Mask1 = get_input2<PrimitiveObject>("Mask1");
            Mask1->verts.resize(w1 * h1);
            Mask1->userData().set2("w", w1);
            Mask1->userData().set2("h", h1);
            if(maskmode1 == "R"){
                Mask1->verts.add_attr<float>("alpha");
                for(int i = 0;i < Mask1->size();i ++){
                    Mask1->verts.attr<float>("alpha")[i] = Mask1->verts[i][0];
                }
                alpha1 = Mask1->verts.attr<float>("alpha");
            }
            if(maskmode1 == "G"){
                Mask1->verts.add_attr<float>("alpha");
                for(int i = 0;i < Mask1->size();i ++){
                    Mask1->verts.attr<float>("alpha")[i] = Mask1->verts[i][1];
                }
                alpha1 = Mask1->verts.attr<float>("alpha");
            }
            if(maskmode1 == "B"){
                Mask1->verts.add_attr<float>("alpha");
                for(int i = 0;i < Mask1->size();i ++){
                    Mask1->verts.attr<float>("alpha")[i] = Mask1->verts[i][2];
                }
                alpha1 = Mask1->verts.attr<float>("alpha");
            }
            if(maskmode1 == "A"){
                if(Mask1->verts.has_attr("alpha")){
                    alpha1 = Mask1->verts.attr<float>("alpha");
                }
                else{
                    Mask1->verts.add_attr<float>("alpha");
                    for (int i = 0; i < h1; i++) {
                        for (int j = 0; j < w1; j++) {
                            Mask1->verts.attr<float>("alpha")[i * w1 + j] = 1;
                        }
                    }
                    alpha1 = Mask1->verts.attr<float>("alpha");
                }
            }
        }
        if(has_input("Mask2")) {
            auto Mask2 = get_input2<PrimitiveObject>("Mask2");
            Mask2->verts.resize(w1 * h1);
            Mask2->userData().set2("w", w1);
            Mask2->userData().set2("h", h1);
            if(maskmode2 == "R"){
                Mask2->verts.add_attr<float>("alpha");
                for(int i = 0;i < Mask2->size();i++){
                    Mask2->verts.attr<float>("alpha")[i] = Mask2->verts[i][0];
                }
                alpha2 = Mask2->verts.attr<float>("alpha");
            }
            if(maskmode2 == "G"){
                Mask2->verts.add_attr<float>("alpha");
                for(int i = 0;i < Mask2->size();i++){
                    Mask2->verts.attr<float>("alpha")[i] = Mask2->verts[i][1];
                }
                alpha2 = Mask2->verts.attr<float>("alpha");
            }
            if(maskmode2 == "B"){
                Mask2->verts.add_attr<float>("alpha");
                for(int i = 0;i < Mask2->size();i++){
                    Mask2->verts.attr<float>("alpha")[i] = Mask2->verts[i][2];
                }
                alpha2 = Mask2->verts.attr<float>("alpha");
            }
            if(maskmode2 == "A"){
                if(Mask2->verts.has_attr("alpha")){
                    alpha2 = Mask2->verts.attr<float>("alpha");
                }
                else{
                    Mask2->verts.add_attr<float>("alpha");
                    for (int i = 0; i < h1; i++) {
                        for (int j = 0; j < w1; j++) {
                            Mask2->verts.attr<float>("alpha")[i * w1 + j] = 1;
                        }
                    }
                    alpha2 = Mask2->verts.attr<float>("alpha");
                }
            }
        }
        if(compmode == "Over") {
//            for (auto a = 0; a < h1 * w1; a++){
//                int i = a / w1;
//                int j = a % w1;
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    vec3f c = rgb1 * l1 + rgb2 * ((l1 != 1 && l2 != 0) ? std::min((1 - l1), l2) : 0);
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = ((l1 != 0 || l2 != 0) ? zeno::max(l2, l1) : 0);
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
                    vec3f c = rgb2 * l2 + rgb1 * ((l2!=1 && l1!=0)? std::min((1-l2),l1) : 0);
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = ((l1!=0 || l2!=0)? zeno::max(l2,l1): 0);
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
                    vec3f c = rgb1 * ((l1 != 0 && l2 != 0) ? l1 : 0) + rgb2 * ((l1 == 0) && (l2 != 0) ? l2 : 0);
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = (l1 !=0 && l2 !=0)? l1 : l2;
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
                    vec3f c = rgb1 * ((l1 != 0) && (l2 != 0) ? l1 : 0);
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = (l1 !=0 && l2 !=0)? l1 : 0;
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
                    vec3f c = rgb1 * ((l1 != 0) && (l2 == 0) ? l1 : 0);
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = (l1 != 0 && l2 == 0)? l1 : 0;
                }
            }
        }
        if(compmode == "Screen"){
            for (int i = 0; i < h1; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb1 = image1->verts[i * w1 + j];
                    vec3f rgb2 = image2->verts[i * w1 + j];
                    float l = zeno::min(zeno::min(image1->verts[i * w1 + j][0],image1->verts[i * w1 + j][1]),image1->verts[i * w1 + j][2]);
                    float l1 = alpha1[i * w1 + j];
                    float l2 = alpha2[i * w1 + j];
                    vec3f c = rgb2 * l2 + rgb2 * ((l1!=0 && l2!=0)? l: 0);
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = l2;
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
                    vec3f c = rgb2 * l2 + rgb1 * l1;
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = zeno::clamp(l1 + l2, 0, 1);
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
                    vec3f c = rgb1 * l1 - rgb2 * l2 ;
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = zeno::clamp(l1 + l2, 0, 1);
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
                    vec3f c = rgb1 * l1 * rgb2 * l2 ;
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = zeno::clamp(l1 + l2, 0, 1);
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
                    vec3f c = rgb1 * l1 / (rgb2 * l2) ;
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = zeno::clamp(l1 + l2, 0, 1);
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
                    vec3f c = abs(rgb1 * l1 - (rgb2 * l2)) ;
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = zeno::clamp(l1 + l2, 0, 1);
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
                    vec3f c = l1 <= l2 ? rgb1 * l1 : rgb2 * l2 ;
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = zeno::clamp(l1 + l2, 0, 1);
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
                    vec3f c = l1 >= l2 ? rgb1 * l1 : rgb2 * l2 ;
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = zeno::clamp(l1 + l2, 0, 1);
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
                    vec3f c = rgb3 * (l1+l2) ;
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = zeno::clamp(l1 + l2, 0, 1);
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
                    vec3f c = (((l1 != 0) && (l2 != 0)) ? rgb3 : rgb1 * l1 + rgb2 * l2) ;
                    image1->verts[i * w1 + j] = zeno::clamp(c, 0, 1);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = zeno::clamp(l1 + l2, 0, 1);
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
                    image1->verts[i * w1 + j] = rgb3 * ((l1 != 0) || (l2 != 0) ? zeno::clamp(l1 + l2, 0, 1) : 0);
                    image1->verts.attr<float>("alpha")[i * w1 + j] = zeno::clamp(l1 + l2, 0, 1);
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
                    image1->verts[i * w1 + j] = rgb3 * ((l1 != 0) || (l2 != 0) ? 0 : zeno::clamp(l1 + l2, 0, 1));
                    image1->verts.attr<float>("alpha")[i * w1 + j] = zeno::clamp(1 - (l1 + l2), 0, 1);
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
        {"enum Over Under Atop Inside Outside Screen Add Subtract Multiply Divide Diff Min Max Average Xor Alpha !Alpha", "Compmode", "Over"},
        {"enum R G B A", "Mask1mode", "R"},
        {"enum R G B A", "Mask2mode", "R"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});


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
    { "image" },
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
    { "image" },
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
    { "image" },
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
    { "image" },
});

struct ImageEdit: INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto size = get_input2<vec2f>("Size");
        auto RGBA = get_input2<std::string>("RGBA");
        auto Gray = get_input2<bool>("Gray");
        auto Invert = get_input2<bool>("Invert");
        auto RGBLevel = get_input2<vec3f>("RGBLevel");
        float R = RGBLevel[0];
        float G = RGBLevel[1];
        float B = RGBLevel[2];
        float L = get_input2<float>("Luminace");
        float ContrastRatio = get_input2<float>("ContrastRatio");
        float Si = get_input2<float>("Saturation");
        auto &ud1 = image->userData();
        int w1 = ud1.get2<int>("w");
        int h1 = ud1.get2<int>("h");
        float H = 0, S = 0, V = 0;
        if(RGBA == "RGBA") {
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
            if(!image->verts.has_attr("alpha")){
                image->verts.add_attr<float>("alpha");
                for(int i = 0;i < image->size();i++){
                    image->verts.attr<float>("alpha")[i] = 1;
                }
            }
        }
        if(RGBA == "RGB") {
            if(image->verts.has_attr("alpha")){
                auto image2 = std::make_shared<PrimitiveObject>();
                image2->verts.resize(w1 * h1);
                image2->userData().set2("isImage", 1);
                image2->userData().set2("w", w1);
                image2->userData().set2("h", h1);
                for (auto i = 0; i < image->verts.size(); i++) {
                    float R1 = image->verts[i][0] * R;
                    float G1 = image->verts[i][1] * G;
                    float B1 = image->verts[i][2] * B;
                    zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                    S = S + (S - 0.5)*(Si-1);
                    V = V + (V - 0.5)*(L-1);
                    zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                    image2->verts[i][0] = R1;
                    image2->verts[i][1] = G1;
                    image2->verts[i][2] = B1;
                }
                image = image2;
            }
            else{
                for (auto i = 0; i < image->verts.size(); i++) {
                    float R1 = image->verts[i][0] * R;
                    float G1 = image->verts[i][1] * G;
                    float B1 = image->verts[i][2] * B;
                    zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                    S = S + (S - 0.5)*(Si-1);
                    V = V + (V - 0.5)*(L-1);
                    zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                    image->verts[i][0] = R1;
                    image->verts[i][1] = G1;
                    image->verts[i][2] = B1;
                }
            }
        }
        if(RGBA == "RA") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = image->verts[i][0] * R;
                float G1 = 0;
                float B1 = 0;
                zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(L-1);
                zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                image->verts[i][0] = R1 ;
                image->verts[i][1] = G1 ;
                image->verts[i][2] = B1 ;
            }
            if(!image->verts.has_attr("alpha")){
                image->verts.add_attr<float>("alpha");
                for(int i = 0;i < image->size();i++){
                    image->verts.attr<float>("alpha")[i] = 1;
                }
            }
        }
        if(RGBA == "GA") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = 0;
                float G1 = G * image->verts[i][1];
                float B1 = 0;
                zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(L-1);
                zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                image->verts[i][0] = R1 ;
                image->verts[i][1] = G1 ;
                image->verts[i][2] = B1 ;
            }
            if(!image->verts.has_attr("alpha")){
                image->verts.add_attr<float>("alpha");
                for(int i = 0;i < image->size();i++){
                    image->verts.attr<float>("alpha")[i] = 1;
                }
            }
        }
        if(RGBA == "BA") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = 0;
                float G1 = 0;
                float B1 = B * image->verts[i][2];
                zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(L-1);
                zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                image->verts[i][0] = R1;
                image->verts[i][1] = G1;
                image->verts[i][2] = B1;
            }
            if(!image->verts.has_attr("alpha")){
                image->verts.add_attr<float>("alpha");
                for(int i = 0;i < image->size();i++){
                    image->verts.attr<float>("alpha")[i] = 1;
                }
            }
        }
        if(RGBA == "R") {
            if(image->verts.has_attr("alpha")){
                auto image2 = std::make_shared<PrimitiveObject>();
                image2->verts.resize(w1 * h1);
                image2->userData().set2("isImage", 1);
                image2->userData().set2("w", w1);
                image2->userData().set2("h", h1);
                for (auto i = 0; i < image->verts.size(); i++) {
                    float R1 = image->verts[i][0] * R;
                    float G1 = 0;
                    float B1 = 0;
                    zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                    S = S + (S - 0.5)*(Si-1);
                    V = V + (V - 0.5)*(L-1);
                    zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                    image2->verts[i][0] = R1;
                    image2->verts[i][1] = G1;
                    image2->verts[i][2] = B1;
                }
                image = image2;
            }
            else{
                for (auto i = 0; i < image->verts.size(); i++) {
                    float R1 = image->verts[i][0] * R;
                    float G1 = 0;
                    float B1 = 0;
                    zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                    S = S + (S - 0.5)*(Si-1);
                    V = V + (V - 0.5)*(L-1);
                    zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                    image->verts[i][0] = R1 ;
                    image->verts[i][1] = G1 ;
                    image->verts[i][2] = B1 ;
                }
            }
        }
        if(RGBA == "G") {
            if(image->verts.has_attr("alpha")){
                auto image2 = std::make_shared<PrimitiveObject>();
                image2->verts.resize(w1 * h1);
                image2->userData().set2("isImage", 1);
                image2->userData().set2("w", w1);
                image2->userData().set2("h", h1);
                for (auto i = 0; i < image->verts.size(); i++) {
                    float R1 = 0;
                    float G1 = image->verts[i][1] * G;
                    float B1 = 0;
                    zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                    S = S + (S - 0.5)*(Si-1);
                    V = V + (V - 0.5)*(L-1);
                    zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                    image2->verts[i][0] = R1;
                    image2->verts[i][1] = G1;
                    image2->verts[i][2] = B1;
                }
                image = image2;
            }
            else{
                for (auto i = 0; i < image->verts.size(); i++) {
                    float R1 = 0;
                    float G1 = image->verts[i][1] * G;
                    float B1 = 0;
                    zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                    S = S + (S - 0.5)*(Si-1);
                    V = V + (V - 0.5)*(L-1);
                    zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                    image->verts[i][0] = R1 ;
                    image->verts[i][1] = G1 ;
                    image->verts[i][2] = B1 ;
                }
            }
        }
        if(RGBA == "B") {
            if(image->verts.has_attr("alpha")){
                auto image2 = std::make_shared<PrimitiveObject>();
                image2->verts.resize(w1 * h1);
                image2->userData().set2("isImage", 1);
                image2->userData().set2("w", w1);
                image2->userData().set2("h", h1);
                for (auto i = 0; i < image->verts.size(); i++) {
                    float R1 = 0;
                    float G1 = 0;
                    float B1 = image->verts[i][2] * B;
                    zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                    S = S + (S - 0.5)*(Si-1);
                    V = V + (V - 0.5)*(L-1);
                    zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                    image2->verts[i][0] = R1;
                    image2->verts[i][1] = G1;
                    image2->verts[i][2] = B1;
                }
                image = image2;
            }
            else{
                for (auto i = 0; i < image->verts.size(); i++) {
                    float R1 = 0;
                    float G1 = 0;
                    float B1 = image->verts[i][2] * B;
                    zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                    S = S + (S - 0.5)*(Si-1);
                    V = V + (V - 0.5)*(L-1);
                    zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                    image->verts[i][0] = R1 ;
                    image->verts[i][1] = G1 ;
                    image->verts[i][2] = B1 ;
                }
            }
        }
        if(RGBA == "A") {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R1 = 1;
                float G1 = 1;
                float B1 = 1;
                zeno::RGBtoHSV(R1, G1, B1, H, S, V);
                S = S + (S - 0.5)*(Si-1);
                V = V + (V - 0.5)*(L-1);
                zeno::HSVtoRGB(H, S, V, R1, G1, B1);
                image->verts[i][0] = R1;
                image->verts[i][1] = G1;
                image->verts[i][2] = B1;
            }
            if (image->verts.has_attr("alpha")) {
                auto &Alpha = image->verts.attr<float>("alpha");
                image->verts.add_attr<float>("alpha");
                image->verts.attr<float>("alpha")=image->verts.attr<float>("alpha");
            }
            else{
                image->verts.add_attr<float>("alpha");
                for(int i = 0;i < w1 * h1;i++){
                    image->verts.attr<float>("alpha")[i] = 1.0;
                }
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
        {"vec2f", "Size", "1,1"},
        {"enum RGBA RGB RA GA BA R G B A", "RGBA", "RGB"},
        {"vec3f", "RGBLevel", "1,1,1"},
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

struct ImageBlur : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
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
        for (auto a = 0; a < image->verts.size(); a++){
            int i = a / w;
            int j = a % w;
            vec3f rgb = image->verts[i * w + j];
            imagecvin.at<cv::Vec3f>(i, j) = {rgb[0], rgb[1], rgb[2]};
        }
        cv::blur(imagecvin,imagecvout,cv::Size(xsize,ysize),cv::Point(-1,-1));
        for (auto a = 0; a < image->verts.size(); a++){
            int i = a / w;
            int j = a % w;
            cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
            image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageBlur, {
    {
        {"image"},
        {"float", "xsize", "10"},
        {"float", "ysize", "10"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
});

struct ImageGaussianBlur : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto kernelsize = get_input2<int>("kernelsize");
        auto sigmaX = get_input2<float>("sigmaX");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        cv::Mat imagecvin(h, w, CV_32FC3);
        cv::Mat imagecvout(h, w, CV_32FC3);
        for (auto a = 0; a < image->verts.size(); a++){
            int i = a / w;
            int j = a % w;
            vec3f rgb = image->verts[i * w + j];
            imagecvin.at<cv::Vec3f>(i, j) = {rgb[0], rgb[1], rgb[2]};
        }
        if(kernelsize%2==0){
            kernelsize += 1;
        }
        cv::GaussianBlur(imagecvin,imagecvout,cv::Size(kernelsize,kernelsize),sigmaX);
        for (auto a = 0; a < image->verts.size(); a++){
            int i = a / w;
            int j = a % w;
            cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
            image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageGaussianBlur, {
    {
        {"image"},
        {"int", "kernelsize", "5"},
        {"float", "sigmaX", "2.0"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
});

struct ImageMedianBlur : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto kernelSize = get_input2<int>("kernelSize");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        cv::Mat imagecvin(h, w, CV_32FC3);
        cv::Mat imagecvout(h, w, CV_32FC3);
        for (auto a = 0; a < image->verts.size(); a++){
            int i = a / w;
            int j = a % w;
            vec3f rgb = image->verts[i * w + j];
            imagecvin.at<cv::Vec3f>(i, j) = {rgb[0], rgb[1], rgb[2]};
        }
        if(kernelSize%2==0){
            kernelSize += 1;
        }
        cv::medianBlur(imagecvin,imagecvout,kernelSize);
        for (auto a = 0; a < image->verts.size(); a++){
            int i = a / w;
            int j = a % w;
            cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
            image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageMedianBlur, {
    {
        {"image"},
        {"int", "kernelSize", "5"},
    },
    {
        {"image"}
    },
    {},
    { "deprecated" },
});

struct ImageBilateralBlur : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto diameter = get_input2<int>("diameter");
        auto sigmaColor = get_input2<float>("sigmaColor");
        auto sigmaSpace = get_input2<float>("sigmaSpace");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        cv::Mat imagecvin(h, w, CV_32FC3);
        cv::Mat imagecvout(h, w, CV_32FC3);
        for (auto a = 0; a < image->verts.size(); a++){
            int i = a / w;
            int j = a % w;
            vec3f rgb = image->verts[i * w + j];
            imagecvin.at<cv::Vec3f>(i, j) = {rgb[0], rgb[1], rgb[2]};
        }
        cv::bilateralFilter(imagecvin,imagecvout, diameter, sigmaColor, sigmaSpace);
        for (auto a = 0; a < image->verts.size(); a++){
            int i = a / w;
            int j = a % w;
            cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
            image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageBilateralBlur, {
    {
        {"image"},
        {"int", "diameter", "10"},
        {"float", "sigmaColor", "75"},
        {"float", "sigmaSpace", "75"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
});
// 自定义卷积核
std::vector<std::vector<float>> createKernel(float blurValue,
                                             float l_blurValue, float r_blurValue,
                                             float t_blurValue, float b_blurValue,
                                             float lt_blurValue, float rt_blurValue,
                                             float lb_blurValue, float rb_blurValue) {
    std::vector<std::vector<float>> kernel;
//    if (isBounded) {
//        kernel = {{blurValue}};
//    }
    kernel = {{lt_blurValue, t_blurValue, rt_blurValue},
              {l_blurValue, blurValue, r_blurValue},
              {lb_blurValue, b_blurValue, rb_blurValue}};
    return kernel;
}
struct CompBlur : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto s = get_input2<float>("strength");
        auto ktop = get_input2<vec3f>("kerneltop");
        auto kmid = get_input2<vec3f>("kernelmid");
        auto kbot = get_input2<vec3f>("kernelbot");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        auto blurredImage = std::make_shared<PrimitiveObject>();
        blurredImage->verts.resize(w * h);
        blurredImage->userData().set2("h", h);
        blurredImage->userData().set2("w", w);
        blurredImage->userData().set2("isImage", 1);
        if(image->has_attr("alpha")){
            blurredImage->verts.attr<float>("alpha") = image->verts.attr<float>("alpha");
        }
        std::vector<std::vector<float>>k = createKernel(kmid[1],kmid[0],kmid[2],ktop[1],kbot[1],ktop[0],ktop[2],kbot[0],kbot[2]);
        int kernelSize = s * k.size();
        int kernelRadius = kernelSize / 2;

// 计算卷积核的中心坐标
        int anchorX = 3 / 2;
        int anchorY = 3 / 2;
        for (int iter = 0; iter < s; iter++) {
            // 对每个像素进行卷积操作
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    float sum0 = 0.0f;
                    float sum1 = 0.0f;
                    float sum2 = 0.0f;
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            int kernelX = x + j - anchorX;
                            int kernelY = y + i - anchorY;

                            if (kernelX >= 0 && kernelX < w && kernelY >= 0 && kernelY < h) {

                                sum0 += image->verts[kernelY * h + kernelX][0] * k[i][j];
                                sum1 += image->verts[kernelY * h + kernelX][1] * k[i][j];
                                sum2 += image->verts[kernelY * h + kernelX][2] * k[i][j];
                            }
                        }
                    }
                    // 将结果赋值给输出图像
                    blurredImage->verts[y * w + x] = {static_cast<float>(sum0), static_cast<float>(sum1),
                                                      static_cast<float>(sum2)};
                }
            }
            image = blurredImage;
        }
        set_output("image", blurredImage);
    }
};

ZENDEFNODE(CompBlur, {
    {
        {"image"},
        {"float", "strength", "5"},
        {"vec3f", "kerneltop", "0.075,0.124,0.075"},
        {"vec3f", "kernelmid", "0.124,0.204,0.124"},
        {"vec3f", "kernelbot", "0.075,0.124,0.075"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
});
static void sobelcv(cv::Mat& src, cv::Mat& dst, int thresholdValue, int maxThresholdValue){

    // 应用Sobel算子计算水平和垂直梯度
    cv::Mat gradX, gradY;
    cv::Sobel(src, gradX, CV_32F, 1, 0);
    cv::Sobel(src, gradY, CV_32F, 0, 1);

    // 计算梯度强度和方向
    cv::Mat gradientMagnitude, gradientDirection;
    cv::cartToPolar(gradX, gradY, gradientMagnitude, gradientDirection, true);

    // 应用阈值
    cv::threshold(gradientMagnitude, dst, thresholdValue, maxThresholdValue, cv::THRESH_BINARY);
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
        if(mode=="sobelcv"){
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2])/3;
                    imagecvin.at<float>(i, j) = var;
                }
            }
            zeno::sobelcv(imagecvin,imagecvout, low_threshold, high_threshold);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
        if(mode=="cannycv"){
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
        {"enum sobel sobelcv cannycv scharr", "mode", "sobel"},
        {"int", "low_threshold", "100"},
        {"int", "high_threshold", "150"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});

struct EdgeDetect_sobel : INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        int thresholdValue = get_input2<int>("thresholdValue");
        int maxValue = get_input2<int>("maxValue");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

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
        if(mode=="sobelcv"){
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            int var = 1;
            for (auto a = 0; a < image->verts.size(); a++){
                int i = a / w;
                int j = a % w;
                vec3f rgb = image->verts[i * w + j];
                var = 255 * (rgb[0] + rgb[1] + rgb[2])/3;
                imagecvin.at<float>(i, j) = var;
            }
            zeno::sobelcv(imagecvin,imagecvout, thresholdValue, maxValue);
            for (auto a = 0; a < image->verts.size(); a++){
                int i = a / w;
                int j = a % w;
                float r = float(imagecvout.at<float>(i, j)) / 255.f;
                image->verts[i * w + j] = {r, r, r};
            }
            set_output("image", image);
        }
        if(mode=="sobelcv2"){
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            int var = 1;
            for (auto a = 0; a < image->verts.size(); a++){
                int i = a / w;
                int j = a % w;
                vec3f rgb = image->verts[i * w + j];
                var = 255 * (rgb[0] + rgb[1] + rgb[2])/3;
                imagecvin.at<float>(i, j) = var;
            }
            cv::Sobel(imagecvin, imagecvout, CV_8U, 1, 1);
            cv::threshold(imagecvin, imagecvout, thresholdValue, maxValue, cv::THRESH_BINARY);
            for (auto a = 0; a < image->verts.size(); a++){
                int i = a / w;
                int j = a % w;
                float r = float(imagecvout.at<float>(i, j)) / 255.f;
                image->verts[i * w + j] = {r, r, r};
            }
            set_output("image", image);
        }
    }
};
ZENDEFNODE(EdgeDetect_sobel, {
    {
        {"image"},
        {"enum sobel sobelcv sobelcv2", "mode", "sobel"},
        {"int", "thresholdValue", "100"},
        {"int", "maxValue", "255"},
    },
    {
        {"image"}
    },
    {},
    { "comp" },
});


struct CompExtractChanel_gray: INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto RGB = get_input2<bool>("RGB");
        auto R = get_input2<bool>("R");
        auto G = get_input2<bool>("G");
        auto B = get_input2<bool>("B");
        auto A = get_input2<bool>("A");
        auto &ud1 = image->userData();
        int w = ud1.get2<int>("w");
        int h = ud1.get2<int>("h");
        auto image2 = std::make_shared<PrimitiveObject>();
        image2->userData().set2("isImage", 1);
        image2->userData().set2("w", w);
        image2->userData().set2("h", h);
        image2->verts.resize(image->size());

        if(RGB){
            for (auto i = 0; i < image->verts.size(); i++) {
                float avr = (image->verts[i][0] + image->verts[i][1] + image->verts[i][2])/3;
                image2->verts[i] = {avr,avr,avr};
            }
        }
        if(R && !RGB) {
            for (auto i = 0; i < image->verts.size(); i++) {
                float R = image->verts[i][0];
                zeno::clamp(image2->verts[i][0] += R,0,1);
                zeno::clamp(image2->verts[i][1] += R,0,1);
                zeno::clamp(image2->verts[i][2] += R,0,1);
            }
        }
        if(G && !RGB) {
            for (auto i = 0; i < image->verts.size(); i++) {
                float G = image->verts[i][1];
                zeno::clamp(image2->verts[i][0] += G,0,1);
                zeno::clamp(image2->verts[i][1] += G,0,1);
                zeno::clamp(image2->verts[i][2] += G,0,1);
            }
        }
        if(B && !RGB) {
            for (auto i = 0; i < image->verts.size(); i++) {
                float B = image->verts[i][2];
                zeno::clamp(image2->verts[i][0] += B,0,1);
                zeno::clamp(image2->verts[i][1] += B,0,1);
                zeno::clamp(image2->verts[i][2] += B,0,1);
            }
        }
        if(A) {
            if (image->verts.has_attr("alpha")) {
                auto &Alpha = image->verts.attr<float>("alpha");
                image2->verts.add_attr<float>("alpha");
                image2->verts.attr<float>("alpha")=image->verts.attr<float>("alpha");
            }
            else{
                image2->verts.add_attr<float>("alpha");
                for(int i = 0;i < w * h;i++){
                    image2->verts.attr<float>("alpha")[i] = 1.0;
                }
            }
        }
        set_output("image", image2);
    }
};
ZENDEFNODE(CompExtractChanel_gray, {
    {
        {"image"},
        {"bool", "RGB", "0"},
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

struct CompExtractChanel : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto RGB = get_input2<bool>("RGB");
        auto R = get_input2<bool>("R");
        auto G = get_input2<bool>("G");
        auto B = get_input2<bool>("B");
        auto A = get_input2<bool>("A");
        auto &ud1 = image->userData();
        int w = ud1.get2<int>("w");
        int h = ud1.get2<int>("h");
        auto image2 = std::make_shared<PrimitiveObject>();
        image2->userData().set2("isImage", 1);
        image2->userData().set2("w", w);
        image2->userData().set2("h", h);
        image2->verts.resize(image->size());
        if(RGB){
            for (auto i = 0; i < image->verts.size(); i++) {
                image2->verts[i][0] = image->verts[i][0];
                image2->verts[i][1] = image->verts[i][1];
                image2->verts[i][2] = image->verts[i][2];
            }
        }
        if(R && !RGB) {
            for (auto i = 0; i < image->verts.size(); i++) {
                image2->verts[i][0] = image->verts[i][0];
            }
        }
        if(G && !RGB) {
            for (auto i = 0; i < image->verts.size(); i++) {
                image2->verts[i][1] = image->verts[i][1];
            }
        }
        if(B && !RGB) {
            for (auto i = 0; i < image->verts.size(); i++) {
                image2->verts[i][2] = image->verts[i][2];
            }
        }
        if(A) {
            if (image->verts.has_attr("alpha")) {
                auto &Alpha = image->verts.attr<float>("alpha");
                image2->verts.add_attr<float>("alpha");
                image2->verts.attr<float>("alpha")=image->verts.attr<float>("alpha");
            }
            else{
                image2->verts.add_attr<float>("alpha");
                for(int i = 0;i < w * h;i++){
                    image2->verts.attr<float>("alpha")[i] = 1.0;
                }
            }
        }
        set_output("image", image2);
    }
};
ZENDEFNODE(CompExtractChanel, {
    {
        {"image"},
        {"bool", "RGB", "0"},
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
        auto mode = get_input2<std::string>("mode");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        vec3i lb = get_input2<vec3i>("low_threshold");
        vec3i ub = get_input2<vec3i>("high_threshold");

        auto A11 = std::make_shared<PrimitiveObject>();
        A11->verts.resize(image->size());
        A11->userData().set2("isImage", 1);
        A11->userData().set2("w", w);
        A11->userData().set2("h", h);
        A11->verts.add_attr<float>("alpha");
        for(int i = 0;i < image->size();i++){
            A11->verts.attr<float>("alpha")[i] = 1;
        }
        std::vector<float> &Alpha = A11->verts.attr<float>("alpha");
        if(image->verts.has_attr("alpha")){
            A11->verts.attr<float>("alpha") = image->verts.attr<float>("alpha");
        }

        if(mode == "Transparent"){
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    if(((lb[0] <= 255 * image->verts[i * w + j][0]) && (255 * image->verts[i * w + j][0] <= ub[0])) &&
                       ((lb[1] <= 255 * image->verts[i * w + j][1]) && (255 * image->verts[i * w + j][1] <= ub[1])) &&
                       ((lb[2] <= 255 * image->verts[i * w + j][2]) && (255 * image->verts[i * w + j][2] <= ub[2]))){
                        A11->verts[i * w + j] = image->verts[i * w + j];
                    }
                    else{

                        A11->verts.attr<float>("alpha")[i * w + j] = 0;
                    }
                }
            }
        }
        else if(mode == "Black"){
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    if(((lb[0] <= 255 * image->verts[i * w + j][0]) && (255 * image->verts[i * w + j][0] <= ub[0])) &&
                       ((lb[1] <= 255 * image->verts[i * w + j][1]) && (255 * image->verts[i * w + j][1] <= ub[1])) &&
                       ((lb[2] <= 255 * image->verts[i * w + j][2]) && (255 * image->verts[i * w + j][2] <= ub[2]))){
                        A11->verts[i * w + j] = image->verts[i * w + j];
                    }
                    else{
                        A11->verts[i * w + j] = {0,0,0};
                    }
                }
            }
        }
        else if(mode == "White"){
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    if(((lb[0] <= 255 * image->verts[i * w + j][0]) && (255 * image->verts[i * w + j][0] <= ub[0])) &&
                       ((lb[1] <= 255 * image->verts[i * w + j][1]) && (255 * image->verts[i * w + j][1] <= ub[1])) &&
                       ((lb[2] <= 255 * image->verts[i * w + j][2]) && (255 * image->verts[i * w + j][2] <= ub[2]))){
                        A11->verts[i * w + j] = image->verts[i * w + j];
                    }
                    else{
                        A11->verts[i * w + j] = {1,1,1};
                    }
                }
            }
        }
        set_output("image", A11);
    }
};
ZENDEFNODE(ImageInRange, {
    {
        {"image"},
        {"enum Transparent Black White", "mode", "Transparent"},
        {"vec3i", "high_threshold", "255,255,255"},
        {"vec3i", "low_threshold", "0,0,0"},
    },
    {
        {"image"},
    },
    {},
    { "image" },
});

struct ImageInRange_black : INode {
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

ZENDEFNODE(ImageInRange_black, {
    {
        {"image"},
        {"vec3i", "high_threshold", "255,255,255"},
        {"vec3i", "low_threshold", "0,0,0"},
    },
    {
        {"image"},
    },
    {},
    { "deprecated" },
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
struct CreateImage : INode {
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
            image->verts[i] = {zeno::clamp(RGB[0]/255, 0, 255),zeno::clamp(RGB[1]/255, 0, 255),zeno::clamp(RGB[2]/255, 0, 255)};
        }
        set_output("image", image);
    }
};

ZENDEFNODE(CreateImage, {
    {
        {"vec3f", "RGB", "255,255,255"},
        {"int", "width", "1024"},
        {"int", "height", "1024"},
    },
    {
        {"image"}
    },
    {},
    { "create" },
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
    { "image" },
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
    { "image" },
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
    {"image"},
});

/* 将灰度图像转换为法线贴图 */
struct ImageToNormalMap : INode {
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
ZENDEFNODE(ImageToNormalMap, {
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
    { "image" },
});

struct ImageGray : INode {
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
ZENDEFNODE(ImageGray, {
    {
        {"image"},
        {"enum average luminace", "mode", "average"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
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
    {"image"},
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
    {"image"},
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
    {"image"},
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
    {"image"},
});

struct ReadImageFromVideo : INode {
    virtual void apply() override {
        auto path = get_input2<std::string>("path");
        auto frame = get_input2<int>("frame");
        std::string native_path = std::filesystem::u8path(path).string();
        cv::VideoCapture videoCapture(native_path.c_str());
        int w = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));

        videoCapture.set(cv::CAP_PROP_POS_FRAMES, frame);
        cv::Mat frameimage(h, w, CV_32FC3);
        bool success = videoCapture.read(frameimage);


        auto image = std::make_shared<PrimitiveObject>();
        image->verts.resize(w * h);

        image->userData().set2("isImage", 1);
        image->userData().set2("w", w);
        image->userData().set2("h", h);
//        zeno::log_info("w:{},h:{}",w,h);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                cv::Vec3f rgb = frameimage.at<cv::Vec3b>(i, j);
                image->verts[(h-i-1) * w + j] = {rgb[0]/255, rgb[1]/255, rgb[2]/255};
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ReadImageFromVideo, {
    {
        {"readpath", "path"},
        {"int", "frame", "1"},
    },
    {
        {"PrimitiveObject", "image"},
    },
    {},
    { "image" },
});
// 计算图像的梯度
void computeGradient(std::shared_ptr<PrimitiveObject> & image, std::vector<std::vector<float>>& gradientX, std::vector<std::vector<float>>& gradientY) {
    auto &ud = image->userData();
    int height  = ud.get2<int>("h");
    int width = ud.get2<int>("w");

    gradientX.resize(height, std::vector<float>(width));
    gradientY.resize(height, std::vector<float>(width));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x > 0 && x < width - 1) {
                gradientX[y][x] = (image->verts[y * width + x + 1][0] - image->verts[y * width + x  - 1])[0] / 2.0f;
            } else {
                gradientX[y][x] = 0.0f;
            }
            if (y > 0 && y < height - 1) {
                gradientY[y][x] = (image->verts[(y+1) * width + x][0] - image->verts[(y - 1) * width + x])[0] / 2.0f;
            } else {
                gradientY[y][x] = 0.0f;
            }
        }
    }
}

// 计算图像的曲率
void computeCurvature(const std::vector<std::vector<float>>& gradientX, const std::vector<std::vector<float>>& gradientY, std::vector<std::vector<float>>& curvature) {
    int height = gradientX.size();
    int width = gradientX[0].size();

    curvature.resize(height, std::vector<float>(width));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float dx = gradientX[y][x];
            float dy = gradientY[y][x];
            float dxx = 0.0f;
            float dyy = 0.0f;
            float dxy = 0.0f;

            if (x > 0 && x < width - 1) {
                dxx = gradientX[y][x + 1] - 2.0f * dx + gradientX[y][x - 1];
            }

            if (y > 0 && y < height - 1) {
                dyy = gradientY[y + 1][x] - 2.0f * dy + gradientY[y - 1][x];
            }

            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                dxy = (gradientX[y + 1][x + 1] - gradientX[y + 1][x - 1] - gradientX[y - 1][x + 1] + gradientX[y - 1][x - 1]) / 4.0f;
            }

            curvature[y][x] = (dxx * dyy - dxy * dxy) / ((dxx + dyy) * (dxx + dyy) + 1e-6f);
        }
    }
}
struct CompCurvature: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input<PrimitiveObject>("image");
        auto threshold = get_input2<float>("threshold");
        auto channel = get_input2<std::string>("channel");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        cv::Mat imagecvgray(h, w, CV_32F);
        cv::Mat imagecvcurvature(h, w, CV_32F);
        if(channel == "R"){
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    imagecvgray.at<float>(i, j) = rgb[0];
                }
            }
        }
        else if(channel == "G"){
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    imagecvgray.at<float>(i, j) = rgb[1];
                }
            }
        }
        else if(channel == "B"){
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    imagecvgray.at<float>(i, j) = rgb[2];
                }
            }
        }
        // 计算图像的梯度
        cv::Mat dx, dy;
        cv::Sobel(imagecvgray, dx, CV_32F, 1, 0);
        cv::Sobel(imagecvgray, dy, CV_32F, 0, 1);
        // 计算梯度的二阶导数
        cv::Mat dxx, dyy, dxy;
        cv::Sobel(dx, dxx, CV_32F, 1, 0);
        cv::Sobel(dy, dyy, CV_32F, 0, 1);
        cv::Sobel(dx, dxy, CV_32F, 0, 1);
        // 计算曲率
        imagecvcurvature = (dxx.mul(dyy) - dxy.mul(dxy)) / ((dxx + dyy).mul(dxx + dyy));
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                float cur = imagecvcurvature.at<float>(i, j);
                if(cur > threshold){
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
ZENDEFNODE(CompCurvature, {
    {
        {"image"},
        {"float","threshold","0"},
        {"enum R G B","channel","R"}
    },
    {
        {"image"},
    },
    {},
    {"Comp"},
});

struct ImageExtractFeature_ORB : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
//        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        cv::Mat imagecvin(h, w, CV_8UC3);
        cv::Mat imagecvgray(h, w, CV_8U);
        cv::Mat imagecvdetect(h, w, CV_8U);
        cv::Mat imagecvout(h, w, CV_8UC3);
        std::vector<cv::KeyPoint> keypoints;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                vec3f rgb = image->verts[i * w + j];
//                imagecvin.at<uchar>(i, j) = 255 * (rgb[0]+rgb[1]+rgb[2])/3;
                imagecvin.at<cv::Vec3b>(i, j) = (255 * rgb[0],255 * rgb[1],255 * rgb[2]);
            }
        }
        cv::cvtColor(imagecvin, imagecvgray, cv::COLOR_BGR2GRAY);
//        imagecvin.convertTo(imagecvgray, CV_8U);
        orb->detect(imagecvgray, keypoints);
        zeno::log_info("orb->detect (imagecvin keypoints:{})",keypoints.size());
        orb->compute(imagecvgray, keypoints, imagecvdetect);
        zeno::log_info("orb->compute(imagecvin, keypoints, imagecvout)");
//        cv::drawKeypoints(imagecvin, keypoints, imagecvout, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

        cv::drawKeypoints(imagecvin, keypoints, imagecvout, cv::Scalar(0, 0, 255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
        zeno::log_info("cv::drawKeypoints(imagecvin, keypoints, imagecvout, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);");
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
//                cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
//                image->verts[i * w + j] = {rgb/255, rgb/255, rgb/255};
//                image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
                image->verts[i * w + j][0] = imagecvin.at<cv::Vec3b>(i, j)[0] ;
                image->verts[i * w + j][1] = imagecvin.at<cv::Vec3b>(i, j)[1] ;
                image->verts[i * w + j][2] = imagecvin.at<cv::Vec3b>(i, j)[2] ;

            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageExtractFeature_ORB, {
    {
        {"image"},
    },
    {
        {"image"},
    },
    {},
    { "image" },
});

struct ImageExtractFeature_SIFT : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
//        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
//        cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
        cv::Mat imagecvin(h, w, CV_8U);
        cv::Mat imagecvout(h, w, CV_8U);
        std::vector<cv::KeyPoint> keypoints;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                vec3f rgb = image->verts[i * w + j];
                imagecvin.at<cv::Vec3f>(i, j) = {rgb[0], rgb[1], rgb[2]};
            }
        }
//        zeno::log_info("imageftok");
//        feature_detector->setContrastThreshold(0.03);  // 设置特征点阈值为0.03
//        sift->setEdgeThreshold(5.0);       // 设置关键点间距阈值为5.0
//        sift->detectAndCompute(imagecvin, cv::noArray(), keypoints, imagecvout);

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                cv::Vec3f rgb = imagecvout.at<cv::Vec3b>(i, j);
                image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageExtractFeature_SIFT, {
    {
        {"image"},
    },
    {
        {"PrimitiveObject", "image"},
    },
    {},
    { "image" },
});

}
}