#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
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
    { "deprecated" },
});



template <class T>
static T BlendMode(const float &alpha1, const float &alpha2, const T& rgb1, const T& rgb2, const vec3f opacity, std::string compmode)
{
        if(compmode == std::string("Copy")) {//copy and over is different!
                T value = rgb1 * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Over")) {
                T value = (rgb1 + rgb2 * (1 - alpha1)) * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Under")) {
                T value = (rgb2 + rgb1 * (1 - alpha2)) * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Atop")) {
                T value = (rgb1 * alpha2 + rgb2 * (1 - alpha1)) * opacity[0] + rgb2 * (1 - opacity[0]);
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
                T value = (rgb1 * (1 - alpha2) + rgb2 * (1 - alpha1)) * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Add")) {
                T value = (rgb1 + rgb2) * opacity[0] + rgb2 * (1 - opacity[0]);//clamp?
                return value;
        }
        else if(compmode == std::string("Subtract")) {
                T value = (rgb2 - rgb1) * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Multiply")) {
                T value = rgb1 * rgb2 * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Max(Lighten)")) {
                T value = zeno::max(rgb1, rgb2) * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Min(Darken)")) {
                T value = zeno::min(rgb1, rgb2) * opacity[0] + rgb2 * (1 - opacity[0]);
                return value;
        }
        else if(compmode == std::string("Screen")) {//A+B-AB if A and B between 0-1, else A if A>B else B
                    T value = (1 - (1 - rgb2) * (1 - rgb1)) * opacity[0] + rgb2 * (1 - opacity[0]);//only care 0-1!
                    return value;
        }
        else if(compmode == std::string("Difference")) {
                    T value = zeno::abs(rgb1 - rgb2) * opacity[0] + rgb2 * (1 - opacity[0]);
                    return value;
        }
        else if(compmode == std::string("Average")) {
                    T value = (rgb1 + rgb2) / 2 * opacity[0] + rgb2 * (1 - opacity[0]);
                    return value;
        }
        return T(0);
}

static zeno::vec3f BlendModeV(const float &alpha1, const float &alpha2, const vec3f& rgb1, const vec3f& rgb2, const vec3f opacity, std::string compmode)
{
        if(compmode == std::string("Overlay")) {
                    vec3f value;
                    for (int k = 0; k < 3; k++) {
                        if (rgb2[k] < 0.5) {
                            value[k] = 2 * rgb1[k] * rgb2[k];
                        } else {
                            value[k] = 1 - 2 * (1 - rgb1[k]) * (1 - rgb2[k]);//screen
                        }
                    }
                    value = value * opacity[0] + rgb2 * (1 - opacity[0]);
                    return value;
        }
        else if(compmode == std::string("SoftLight")) {
                    vec3f value;
                    for (int k = 0; k < 3; k++) {
                        if (rgb1[k] < 0.5) {
                            value[k] = 2 * rgb1[k] * rgb2[k] + rgb2[k] * rgb2[k] * (1 - 2 * rgb1[k]);
                        } else {
                            value[k] = 2 * rgb2[k] * (1 - rgb1[k]) + sqrt(rgb2[k]) * (2 * rgb1[k] - 1);
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
                            value[k] = rgb2[k] / rgb1[k];
                        }
                    }
                    value = value * opacity[0] + rgb2 * (1 - opacity[0]);
                    return value;
        }
        return zeno::vec3f(0);
}

struct Blend: INode {
    virtual void apply() override {//todo:: add blend scope RGBA and Premultiplied / Alpha Blending(https://github.com/jamieowen/glsl-blend/issues/6)
        auto blend = get_input<PrimitiveObject>("Foreground");
        auto base = get_input<PrimitiveObject>("Background");
        auto maskopacity = get_input2<float>("Mask Opacity");

        auto compmode = get_input2<std::string>("Blending Mode");
        //auto alphablend = get_input2<std::string>("Alpha Blending");
        auto alphamode = get_input2<std::string>("Alpha Mode");
        auto opacity1 = get_input2<float>("Foreground Opacity");
        auto opacity2 = get_input2<float>("Background Opacity");

        auto &ud1 = blend->userData();
        int w1 = ud1.get2<int>("w");
        int h1 = ud1.get2<int>("h");
        int imagesize = w1 * h1;
        auto mask = std::make_shared<PrimitiveObject>();
        if(has_input("Mask")) {
            mask = get_input<PrimitiveObject>("Mask");

        }
        else {
            mask->verts.resize(w1*h1);
            mask->userData().set2("isImage", 1);
            mask->userData().set2("w", w1);
            mask->userData().set2("h", h1);
            for (int i = 0; i < imagesize; i++) {
                    mask->verts[i] = {maskopacity,maskopacity,maskopacity};
            }
        }
        auto image2 = std::make_shared<PrimitiveObject>();
        image2->userData().set2("isImage", 1);
        image2->userData().set2("w", w1);
        image2->userData().set2("h", h1);
        image2->verts.resize(imagesize);
        bool alphaoutput =  blend->has_attr("alpha")||base->has_attr("alpha");
        auto &image2alpha = image2->add_attr<float>("alpha");
        auto &blendalpha = blend->has_attr("alpha")?blend->attr<float>("alpha"):std::vector<float>(imagesize, 1.0f);
        auto &basealpha = base->has_attr("alpha")?base->attr<float>("alpha"):std::vector<float>(imagesize, 1.0f);

//todo： rgb1和rgb2大小不同的情况
#pragma omp parallel for
            for (int i = 0; i < imagesize; i++) {
                vec3f rgb1 = blend->verts[i] * opacity1;
                vec3f rgb2 = base->verts[i] * opacity2;
                vec3f opacity = zeno::clamp(mask->verts[i], 0, 1) * maskopacity;
                float alpha1 = zeno::clamp(blendalpha[i], 0, 1);
                float alpha2 = zeno::clamp(basealpha[i], 0, 1);
                if(compmode == "Overlay" || compmode == "SoftLight" || compmode == "Divide"){
                    vec3f c = BlendModeV(alpha1, alpha2, rgb1, rgb2, opacity, compmode);
                    image2->verts[i] = c;
                }
                else{
                    vec3f c = BlendMode<zeno::vec3f>(alpha1, alpha2, rgb1, rgb2, opacity, compmode);
                    image2->verts[i] = c;
                }
            }
            if(alphaoutput) {//如果两个输入 其中一个没有alpha  对于rgb和alpha  alpha的默认值不一样 前者为1 后者为0？
                auto &blendalpha = blend->has_attr("alpha")?blend->attr<float>("alpha"):blend->add_attr<float>("alpha");//只有blendbase都没有alpha 结果才没有
                auto &basealpha = base->has_attr("alpha")?base->attr<float>("alpha"):base->add_attr<float>("alpha");
                //std::string alphablendmode = alphamode == "SameWithBlend" ? compmode : alphamode;
#pragma omp parallel for
                for (int i = 0; i < imagesize; i++) {
                vec3f opacity = zeno::clamp(mask->verts[i], 0, 1) * maskopacity;
                float alpha = BlendMode<float>(zeno::clamp(blendalpha[i], 0, 1), zeno::clamp(basealpha[i], 0, 1), zeno::clamp(blendalpha[i], 0, 1), zeno::clamp(basealpha[i], 0, 1), opacity, alphamode);
                image2alpha[i] = alpha;
                }
            }

        set_output("image", image2);
    }
};

ZENDEFNODE(Blend, {
    {
        {"Foreground"},
        {"Background"},
        {"Mask"},
        {"enum Over Copy Under Atop In Out Screen Add Subtract Multiply Max(Lighten) Min(Darken) Average Difference Overlay SoftLight Divide Xor", "Blending Mode", "Normal"},
        //{"enum IgnoreAlpha SourceAlpha", "Alpha Blending", "Ignore Alpha"}, SUBSTANCE DESIGNER ALPHA MODE
        //{"enum SameWithBlend Over Under Atop In Out Screen Add Subtract Multiply Max(Lighten) Min(Darken) Average Difference Xor", "Alpha Mode", "SameWithBlend"},
        {"enum Over Under Atop In Out Screen Add Subtract Multiply Max(Lighten) Min(Darken) Average Difference Xor", "Alpha Mode", "Over"},
        {"float", "Mask Opacity", "1"},
        {"float", "Foreground Opacity", "1"},
        {"float", "Background Opacity", "1"},
    },
    {
        {"image"}
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

struct CompBlur : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto s = get_input2<int>("strength");
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
            
            blurredImage->verts.add_attr<float>("alpha");
            blurredImage->verts.attr<float>("alpha") = image->verts.attr<float>("alpha");
        }
        std::vector<std::vector<float>>k = createKernel(kmid[1],kmid[0],kmid[2],ktop[1],kbot[1],ktop[0],ktop[2],kbot[0],kbot[2]);
// 计算卷积核的中心坐标
        int anchorX = 3 / 2;
        int anchorY = 3 / 2;
        for (int iter = 0; iter < s; iter++) {
#pragma omp parallel for
            // 对每个像素进行卷积操作
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    float sum0 = 0.0f;
                    float sum1 = 0.0f;
                    float sum2 = 0.0f;
                    if (x == 0 || x == w - 1 || y == 0 || y == h - 1) {
                        sum0 = image->verts[y * w + x][0];
                        sum1 = image->verts[y * w + x][1];
                        sum2 = image->verts[y * w + x][2];
                    } 
                    else
                    {
                        for (int i = 0; i < 3; i++) {
                            for (int j = 0; j < 3; j++) {
                                int kernelX = x + j - anchorX;
                                int kernelY = y + i - anchorY;
                                sum0 += image->verts[kernelY * w + kernelX][0] * k[i][j];
                                sum1 += image->verts[kernelY * w + kernelX][1] * k[i][j];
                                sum2 += image->verts[kernelY * w + kernelX][2] * k[i][j];
                            }
                        }
                    }
                    blurredImage->verts[y * w + x] = {static_cast<float>(sum0),
                                                      static_cast<float>(sum1),
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
        {"int", "strength", "5"},
        {"vec3f", "kerneltop", "0.075,0.124,0.075"},
        {"vec3f", "kernelmid", "0.124,0.204,0.124"},
        {"vec3f", "kernelbot", "0.075,0.124,0.075"},
    },
    {
        {"image"}
    },
    {},
    { "deprecated" },
});

struct ImageExtractChannel : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto channel = get_input2<std::string>("channel");
        auto &ud1 = image->userData();
        int w = ud1.get2<int>("w");
        int h = ud1.get2<int>("h");
        auto image2 = std::make_shared<PrimitiveObject>();
        image2->userData().set2("isImage", 1);
        image2->userData().set2("w", w);
        image2->userData().set2("h", h);
        image2->verts.resize(image->size());
        if(channel == "R") {
            for (auto i = 0; i < image->verts.size(); i++) {
                image2->verts[i] = vec3f(image->verts[i][0]);
            }
        }
        else if(channel == "G") {
            for (auto i = 0; i < image->verts.size(); i++) {
                image2->verts[i] = vec3f(image->verts[i][1]);
            }
        }
        else if(channel == "B") {
            for (auto i = 0; i < image->verts.size(); i++) {
                image2->verts[i] = vec3f(image->verts[i][2]);
            }
        }
        else if(channel == "A") {
            if (image->verts.has_attr("alpha")) {
                auto &attr = image->verts.attr<float>("alpha");
                for(int i = 0; i < w * h; i++){
                    image2->verts[i] = {attr[i], attr[i], attr[i]};
                }
            }
            else{
                throw zeno::makeError("image have no alpha channel");
            }
        }
        set_output("image", image2);
    }
};
ZENDEFNODE(ImageExtractChannel, {
    {
        {"image"},
        {"enum R G B A", "channel", "R"},
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
        int nx = ud.has("nx")?ud.get2<int>("nx"):ud.get2<int>("w");
        int ny = ud.has("ny")?ud.get2<int>("ny"):ud.get2<int>("h");
        auto attrName = get_input2<std::string>("attrName");
        auto remapRange = get_input2<vec2f>("RemapRange");
        auto remap = get_input2<bool>("Remap");
        auto image = std::make_shared<PrimitiveObject>();
        auto attributesType = get_input2<std::string>("AttributesType");

        image->resize(nx * ny);
        image->userData().set2("isImage", 1);
        image->userData().set2("w", nx);
        image->userData().set2("h", ny);

        std::visit([&](auto attrty) {
            using T = decltype(attrty);
            if (!prim->verts.has_attr(attrName)) {
                zeno::log_error("No such attribute '{}' in prim", attrName);
                return;
            }
            auto &attr = prim->verts.attr<T>(attrName);
            auto minresult = zeno::parallel_reduce_array<T>(attr.size(), attr[0], [&] (size_t i) -> T { return attr[i]; },
            [&] (T i, T j) -> T { return zeno::min(i, j); });
            auto maxresult = zeno::parallel_reduce_array<T>(attr.size(), attr[0], [&] (size_t i) -> T { return attr[i]; },
            [&] (T i, T j) -> T { return zeno::max(i, j); });

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
        }, enum_variant<std::variant<float, vec3f>>(array_index({"float", "vec3f"}, attributesType)));
        set_output("image", image);
    }
};

ZENDEFNODE(CompImport, {
    {
        {"prim"},
        {"string", "attrName", ""},
        {"bool", "Remap", "0"},
        {"vec2f", "RemapRange", "0, 1"},
        {"enum float vec3f", "AttributesType", "float"},
    },
    {
        {"image"},
    },
    {},
    { "comp" },
});

struct CompMixChanel : INode {
    virtual void apply() override {
        auto R = get_input<PrimitiveObject>("R");
        auto G = get_input<PrimitiveObject>("G");
        auto B = get_input<PrimitiveObject>("B");
        auto &ud = R->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        for(int i = 0;i < R->size();i++){
            R->verts[i][1] = G->verts[i][1];
            R->verts[i][2] = B->verts[i][2];
        }
        set_output("image", R);
    }
};

ZENDEFNODE(CompMixChanel, {
    {
        {"R"},
        {"G"},
        {"B"},
    },
    {
        {"image"},
    },
    {},
    { "comp" },
});
}
}