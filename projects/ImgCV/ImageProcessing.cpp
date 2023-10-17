#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/NumericObject.h>
#include <random>
#include <zeno/utils/scope_exit.h>
#include <stdexcept>
#include <cmath>
#include <zeno/utils/log.h>
#include <opencv2/opencv.hpp>


using namespace cv;

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
        //if(image->has_attr("alpha")){
            //image2->verts.add_attr<float>("alpha");
        //}

        float scaleX = static_cast<float>(w) / width;
        float scaleY = static_cast<float>(h) / height;

        for (auto a = 0; a < image->verts.size(); a++){
            int x = a / w;
            int y = a % w;
            int srcX = static_cast<int>(x * scaleX);
            int srcY = static_cast<int>(y * scaleY);
            image2->verts[y * width + x] = image->verts[srcY * w + srcX];
            //image2->verts.attr<float>("alpha")[y * width + x] = image->verts.attr<float>("alpha")[srcY * w + srcX];
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

    double radians = angle * 3.14159 / 180.0;
    int width = src->userData().get2<int>("w");
    int height = src->userData().get2<int>("h");

    int centerX = width / 2;
    int centerY = height / 2;

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

                int srcX = static_cast<int>((x - rotatedWidth / 2) * cos(-radians) - (y - rotatedHeight / 2) * sin(-radians) + centerX);
                int srcY = static_cast<int>((x - rotatedWidth / 2) * sin(-radians) + (y - rotatedHeight / 2) * cos(-radians) + centerY);

                if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
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
                int srcX = static_cast<int>((x - rotatedWidth / 2) * cos(-radians) - (y - rotatedHeight / 2) * sin(-radians) + centerX);
                int srcY = static_cast<int>((x - rotatedWidth / 2) * sin(-radians) + (y - rotatedHeight / 2) * cos(-radians) + centerY);

                if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                    dst->verts[y * rotatedWidth + x] = src->verts[srcY * width + srcX] ;
                    dst->verts.attr<float>("alpha")[y * rotatedWidth + x] = 1;
                }
            }
        }
    }
    for (int y = 0; y < rotatedHeight; ++y) {
        for (int x = 0; x < rotatedWidth; ++x) {
            int srcX = static_cast<int>((x - rotatedWidth / 2) * cos(-radians) - (y - rotatedHeight / 2) * sin(-radians) + centerX);
            int srcY = static_cast<int>((x - rotatedWidth / 2) * sin(-radians) + (y - rotatedHeight / 2) * cos(-radians) + centerY);

            if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
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
    { "image" },
});

float gaussian(float x, float sigma) {
    return exp(-(x * x) / (2 * sigma * sigma));
}
void gaussian_filter(std::shared_ptr<PrimitiveObject> &image, std::shared_ptr<PrimitiveObject> &imagetmp, int width, int height, int sigma) {

    int size = (int)(2 * sigma + 1);
    if (size % 2 == 0) {
        size++;
    }

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

    delete[] kernel;
}

// MedianBlur
void MedianBlur(std::shared_ptr<PrimitiveObject> &image, std::shared_ptr<PrimitiveObject> &imagetmp, int width, int height, int kernel_size) {

    using kernel = std::tuple<float, float, float>;
    kernel n = {0, 0, 0};
    std::vector<kernel> kernel_values(kernel_size * kernel_size);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            int current_value0 = image->verts[y * width + x][0];
            int current_value1 = image->verts[y * width + x][1];
            int current_value2 = image->verts[y * width + x][2];

            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int px = x - kernel_size / 2 + kx;
                    int py = y - kernel_size / 2 + ky;

                    if (px < 0 || px >= width || py < 0 || py >= height) {
                        kernel_values[ky * kernel_size + kx] = {current_value0,current_value1,current_value2};
                    }
                    else {
                        kernel_values[ky * kernel_size + kx] = {image->verts[py * width + px][0],image->verts[py * width + px][1],image->verts[py * width + px][2]};
                    }
                }
            }

            std::sort(kernel_values.begin(), kernel_values.end());
            float new_value0 = std::get<0>(kernel_values[kernel_size * kernel_size / 2]);
            float new_value1 = std::get<1>(kernel_values[kernel_size * kernel_size / 2]);
            float new_value2 = std::get<2>(kernel_values[kernel_size * kernel_size / 2]);

            imagetmp->verts[y * width + x] = {new_value0,new_value1,new_value2};
        }
    }
    image = imagetmp;
}

float bilateral(float src, float dst, float sigma_s, float sigma_r) {
    return gaussian(src - dst, sigma_s) * gaussian(abs(src - dst), sigma_r);
}

void bilateralFilter(std::shared_ptr<PrimitiveObject> &image, std::shared_ptr<PrimitiveObject> &imagetmp, int width, int height, float sigma_s, float sigma_r) {
    int k = ceil(3 * sigma_s);
    float* tmp = new float[width * height];
    for (int i = k; i < height-k; i++) {
        for (int j = k; j < width-k; j++) {
            float sum0 = 0, sum1 = 0, sum2 = 0;
            float wsum0 = 0,wsum1 = 0,wsum2 = 0;
            for (int m = -k; m <= k; m++) {
                for (int n = -k; n <= k; n++) {
                    float w0 = bilateral(image->verts[i*width+j][0],image->verts[(i+m)*width+j+n][0], sigma_s, sigma_r);
                    float w1 = bilateral(image->verts[i*width+j][1],image->verts[(i+m)*width+j+n][1], sigma_s, sigma_r);
                    float w2 = bilateral(image->verts[i*width+j][2],image->verts[(i+m)*width+j+n][2], sigma_s, sigma_r);

                    sum0 += w0 * image->verts[(i+m)*width+j+n][0];
                    sum1 += w1 * image->verts[(i+m)*width+j+n][1];
                    sum2 += w2 * image->verts[(i+m)*width+j+n][2];

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


///////////////////////////
std::vector<int> boxesForGauss(float sigma, int n)  // standard deviation, number of boxes
{
	auto wIdeal = sqrt((12 * sigma*sigma / n) + 1);  // Ideal averaging filter width
	int wl = floor(wIdeal);
	if (wl % 2 == 0)
		wl--;
	int wu = wl + 2;

	float mIdeal = (12 * sigma*sigma - n * wl*wl - 4 * n*wl - 3 * n) / (-4 * wl - 4);
	int m = round(mIdeal);
	// var sigmaActual = Math.sqrt( (m*wl*wl + (n-m)*wu*wu - n)/12 );

	std::vector<int> sizes(n);
	for (auto i = 0; i < n; i++)
		sizes[i] = i < m ? wl : wu;
	return sizes;
}

void boxBlurH(std::vector<zeno::vec3f>& scl, std::vector<zeno::vec3f>& tcl, int w, int h, int r) {
	float iarr = 1.f / (r + r + 1);
    #pragma omp parallel for
	for (int i = 0; i < h; i++) {
		int ti = i * w, li = ti, ri = ti + r;
		auto fv = scl[ti], lv = scl[ti + w - 1];
		auto val = (r + 1)*fv;
		for (int j = 0; j < r; j++) val += scl[ti + j];
		for (int j = 0; j <= r; j++, ri++, ti++) { val += scl[ri] - fv;   tcl[ti] = val*iarr; }
		for (int j = r + 1; j < w - r; j++, ri++, ti++, li++) { val += scl[ri] - scl[li];   tcl[ti] = val*iarr; }
		for (int j = w - r; j < w; j++, ti++, li++) { val += lv - scl[li];   tcl[ti] = val*iarr; }//border?
	}
}
void boxBlurT(std::vector<zeno::vec3f>& scl, std::vector<zeno::vec3f>& tcl, int w, int h, int r) {
	float iarr = 1.f / (r + r + 1);// radius range on either side of a pixel + the pixel itself
    #pragma omp parallel for
	for (auto i = 0; i < w; i++) {
		int ti = i, li = ti, ri = ti + r * w;
		auto fv = scl[ti], lv = scl[ti + w * (h - 1)];
		auto val = (r + 1)*fv;
		for (int j = 0; j < r; j++) val += scl[ti + j * w];
		for (int j = 0; j <= r; j++, ri+=w, ti+=w) { val += scl[ri] - fv;  tcl[ti] = val*iarr; }
		for (int j = r + 1; j < h - r; j++, ri+=w, ti+=w, li+=w) { val += scl[ri] - scl[li];  tcl[ti] = val*iarr; }
		for (int j = h - r; j < h; j++, ti+=w, li+=w) { val += lv - scl[li];  tcl[ti] = val*iarr; }
	}
}
void boxBlur(std::vector<zeno::vec3f>& scl, std::vector<zeno::vec3f>& tcl, int w, int h, int r) {
    std::swap(scl, tcl);
	boxBlurH(tcl, scl, w, h, r);
	boxBlurT(scl, tcl, w, h, r);
}
void gaussBlur(std::vector<zeno::vec3f> scl, std::vector<zeno::vec3f>& tcl, int w, int h, float sigma, int blurNumber) {
	auto bxs = boxesForGauss(sigma, blurNumber);
	boxBlur(scl, tcl, w, h, (bxs[0] - 1) / 2);
	boxBlur(tcl, scl, w, h, (bxs[1] - 1) / 2);
	boxBlur(scl, tcl, w, h, (bxs[2] - 1) / 2);
    /*for (auto i = 0; i < blurNumber; i++) {
        boxBlur(scl, tcl, w, h, (bxs[i] - 1) / 2);
    }*/
}

struct ImageBlur : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto kernelSize = get_input2<int>("kernelSize");
        auto type = get_input2<std::string>("type");
        auto fastgaussian = get_input2<bool>("Fast Blur(Gaussian)");
        auto sigmaX = get_input2<float>("GaussianSigma");
        auto sigmaColor = get_input2<vec2f>("BilateralSigma")[0];
        auto sigmaSpace = get_input2<vec2f>("BilateralSigma")[1];
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        auto img_out = std::make_shared<PrimitiveObject>();
        img_out->resize(w * h);
        img_out->userData().set2("w", w);
        img_out->userData().set2("h", h);
        img_out->userData().set2("isImage", 1);

        if(type == "Gaussian" && fastgaussian){
            gaussBlur(image->verts, img_out->verts, w, h, sigmaX, 3);
        }
        else{//CV BLUR
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
            if(type == "Box"){
                cv::boxFilter(imagecvin,imagecvout,-1,cv::Size(kernelSize,kernelSize));
            }
            else if(type == "Gaussian"){
                cv::GaussianBlur(imagecvin,imagecvout,cv::Size(kernelSize,kernelSize),sigmaX);
            }
            else if(type == "Median"){
                cv::medianBlur(imagecvin,imagecvout,kernelSize);//kernel size can only be 3/5 when use CV_32FC3
            }
            else if(type == "Bilateral"){
                cv::bilateralFilter(imagecvin,imagecvout, kernelSize, sigmaColor, sigmaSpace);
            }
            else if(type == "Stack"){
                cv::stackBlur(imagecvin,imagecvout,cv::Size(kernelSize, kernelSize));
            }
            else{
                zeno::log_error("ImageBlur: Blur type does not exist");
            }
            for (auto a = 0; a < image->verts.size(); a++){
                int i = a / w;
                int j = a % w;
                cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
                img_out->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
            }
        }
        set_output("image", img_out);
    }
};

ZENDEFNODE(ImageBlur, {
    {
        {"image"},
        {"int", "kernelSize", "5"},
        {"enum Gaussian Box Median Bilateral Stack", "type", "Gaussian"},
        {"float", "GaussianSigma", "1"},//fast gaussian only effect by sigma  等参数分开显示再移开
        {"vec2f", "BilateralSigma", "50,50"},
        {"bool", "Fast Blur(Gaussian)", "1"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
});

struct ImageEditContrast : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        float ContrastRatio = get_input2<float>("ContrastRatio");
        float ContrastCenter = get_input2<float>("ContrastCenter");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        for (auto i = 0; i < image->verts.size(); i++) {
            image->verts[i] = image->verts[i] + (image->verts[i]-ContrastCenter) * (ContrastRatio-1);
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageEditContrast, {
    {
        {"image"},
        {"float", "ContrastRatio", "1"},
        {"float", "ContrastCenter", "0.5"},
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
    {"deprecated"},
});

/* 将灰度图像转换为法线贴图 */
struct ImageToNormalMap : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto strength = get_input2<float>("strength");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        auto InvertR = get_input2<bool>("InvertR");
        auto InvertG = get_input2<bool>("InvertG");
        auto normalmap = std::make_shared<PrimitiveObject>();
        normalmap->verts.resize(w * h);
        normalmap->userData().set2("isImage", 1);
        normalmap->userData().set2("w", w);
        normalmap->userData().set2("h", h);

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int idx = i * w + j;
                if (i == 0 || i == h || j == 0 || j == w) {
                    normalmap->verts[idx] = {0, 0, 1};
                }
            }
        }
        
#pragma omp parallel for
        for (int i = 1; i < h-1; i++) {
            for (int j = 1; j < w-1; j++) {
                float gx = -image->verts[(i - 1) * w + j - 1][0] + image->verts[(i - 1) * w + j + 1][0] 
                - 2.0f * image->verts[i * w + j - 1][0] + 2.0f * image->verts[ i * w + j + 1][0]
                - image->verts[(i + 1) * w + j - 1][0] + image->verts[(i + 1) * w + j + 1][0]; 

                float gy = image->verts[(i - 1) * w + j - 1][0] + 2.0f * image->verts[(i - 1) * w + j][0]
                + image->verts[(i - 1) * w + j + 1][0] - image->verts[(i + 1) * w + j - 1][0]
                - 2.0f * image->verts[(i + 1) * w + j][0] - image->verts[(i + 1) * w + j + 1][0]; 
                
                gx = gx * strength;
                gy = gy * strength;
               vec3f rgb = {gx,gy,1};

                rgb /= length(rgb);
                rgb = normalizeSafe(rgb);
                rgb = 0.5f * (rgb + 1.0f) ;
                if(InvertG){
                    rgb[1] = 1 - rgb[1];
                }
                else if(InvertR){
                    rgb[0] = 1 - rgb[0];
                }
                normalmap->verts[i * w + j] = rgb;
                
                }
            }

        set_output("image", normalmap);
    }
};

ZENDEFNODE(ImageToNormalMap, {
    {
        {"image"},
        {"float", "strength", "10"},
        {"bool", "InvertR", "0"},
        {"bool", "InvertG", "0"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
});

struct ImageGray : INode {//todo
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
    int image_height = image->userData().get2<int>("h");
    int image_width = image->userData().get2<int>("w");
    int kernel_height = kernel.size();
    int kernel_width = kernel[0].size();
    int center_y = kernel_height / 2;
    int center_x = kernel_width / 2;

#pragma omp parallel for
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
                for (int ky = 0; ky < kernel_height; ky++) {
                    for (int kx = 0; kx < kernel_width; kx++) {
                        int image_y = y - center_y + ky;
                        int image_x = x - center_x + kx;
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
                imagetmp->verts[y * image_width + x]= {maxValue0,maxValue1,maxValue2};
            }
        }
        image = imagetmp;
    }
}

void dilateImage(cv::Mat& src, cv::Mat& dst, int kheight, int kwidth, int Strength) {
    cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(kheight, kwidth));
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
#pragma omp parallel for
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
//#pragma omp parallel for
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                vec3f rgb = image->verts[i * w + j];
                imagecvin.at<cv::Vec3f>(i, j) = {rgb[0], rgb[1], rgb[2]};
            }
        }

        cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * kheight + 1, 2 * kwidth + 1),
                                               cv::Point(1, 1));
        cv::erode(imagecvin, imagecvout, kernel,cv::Point(-1, -1), strength);

//#pragma omp parallel for
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


struct ImageColor : INode {
    virtual void apply() override {
        auto image = std::make_shared<PrimitiveObject>();
        auto color = get_input2<vec4f>("Color");
        auto size = get_input2<vec2i>("Size");
        auto balpha = get_input2<bool>("alpha");
        auto vertsize = size[0] * size[1];
        image->verts.resize(vertsize);
        image->userData().set2("isImage", 1);
        image->userData().set2("w", size[0]);
        image->userData().set2("h", size[1]);
        if(balpha){
            auto &alphaAttr = image->verts.add_attr<float>("alpha");
            for (int i = 0; i < vertsize ; i++) {
                image->verts[i] = {zeno::clamp(color[0], 0.0f, 1.0f), zeno::clamp(color[1], 0.0f, 1.0f), zeno::clamp(color[2], 0.0f, 1.0f)};
                alphaAttr[i] = zeno::clamp(color[3], 0.0f, 1.0f);
            }
        }
        else{
            for (int i = 0; i < vertsize ; i++) {
                image->verts[i] = {zeno::clamp(color[0], 0.0f, 1.0f), zeno::clamp(color[1], 0.0f, 1.0f), zeno::clamp(color[2], 0.0f, 1.0f)};
            }
        }
        set_output("image", image);
        
    }
};

ZENDEFNODE(ImageColor, {
    {
        {"vec4f", "Color", "1,1,1,1"},
        {"vec2i", "Size", "1024,1024"},
        {"bool", "alpha", "1"},
    },
    {
        {"image"},
    },
    {},
    { "deprecated" },
});
struct ImageColor2 : INode {
    virtual void apply() override {
        auto image = std::make_shared<PrimitiveObject>();
        auto color = get_input2<vec3f>("Color");
        auto alpha = get_input2<float>("Alpha");
        auto size = get_input2<vec2i>("Size");
        auto balpha = get_input2<bool>("alpha");
        auto vertsize = size[0] * size[1];
        image->verts.resize(vertsize);
        image->userData().set2("isImage", 1);
        image->userData().set2("w", size[0]);
        image->userData().set2("h", size[1]);
        if(balpha){
            auto &alphaAttr = image->verts.add_attr<float>("alpha");
            for (int i = 0; i < vertsize ; i++) {
                image->verts[i] = {zeno::clamp(color[0], 0.0f, 1.0f), zeno::clamp(color[1], 0.0f, 1.0f), zeno::clamp(color[2], 0.0f, 1.0f)};
                alphaAttr[i] = zeno::clamp(alpha, 0.0f, 1.0f);
            }
        }
        else{
            for (int i = 0; i < vertsize ; i++) {
                image->verts[i] = {zeno::clamp(color[0], 0.0f, 1.0f), zeno::clamp(color[1], 0.0f, 1.0f), zeno::clamp(color[2], 0.0f, 1.0f)};
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageColor2, {
    {
        {"vec3f", "Color", "1,1,1"},
        {"float", "Alpha", "1"},
        {"vec2i", "Size", "1024,1024"},
        {"bool", "alpha", "1"},
    },
    {
        {"image"},
    },
    {},
    { "image" },
});


//TODO:: fix sparse convolution noise

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Sparse Convolution Noise
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// std::array<int, 256> perm = {
//     225, 155, 210, 108, 175, 199, 221, 144, 203, 116, 70,  213, 69,  158, 33,  252, 5,   82,  173, 133, 222, 139,
//     174, 27,  9,   71,  90,  246, 75,  130, 91,  191, 169, 138, 2,   151, 194, 235, 81,  7,   25,  113, 228, 159,
//     205, 253, 134, 142, 248, 65,  224, 217, 22,  121, 229, 63,  89,  103, 96,  104, 156, 17,  201, 129, 36,  8,
//     165, 110, 237, 117, 231, 56,  132, 211, 152, 20,  181, 111, 239, 218, 170, 163, 51,  172, 157, 47,  80,  212,
//     176, 250, 87,  49,  99,  242, 136, 189, 162, 115, 44,  43,  124, 94,  150, 16,  141, 247, 32,  10,  198, 223,
//     255, 72,  53,  131, 84,  57,  220, 197, 58,  50,  208, 11,  241, 28,  3,   192, 62,  202, 18,  215, 153, 24,
//     76,  41,  15,  179, 39,  46,  55,  6,   128, 167, 23,  188, 106, 34,  187, 140, 164, 73,  112, 182, 244, 195,
//     227, 13,  35,  77,  196, 185, 26,  200, 226, 119, 31,  123, 168, 125, 249, 68,  183, 230, 177, 135, 160, 180,
//     12,  1,   243, 148, 102, 166, 38,  238, 251, 37,  240, 126, 64,  74,  161, 40,  184, 149, 171, 178, 101, 66,
//     29,  59,  146, 61,  254, 107, 42,  86,  154, 4,   236, 232, 120, 21,  233, 209, 45,  98,  193, 114, 78,  19,
//     206, 14,  118, 127, 48,  79,  147, 85,  30,  207, 219, 54,  88,  234, 190, 122, 95,  67,  143, 109, 137, 214,
//     145, 93,  92,  100, 245, 0,   216, 186, 60,  83,  105, 97,  204, 52};

// template <typename T>
// constexpr T PERM(T x) {
//     return perm[(x)&255];
// }

// #define INDEX(ix, iy, iz) PERM((ix) + PERM((iy) + PERM(iz)))

// std::random_device rd;
// std::default_random_engine engine(rd());
// std::uniform_real_distribution<float> d(0, 1);

// float impulseTab[256 * 4];
// void impulseTabInit() {
//     int i;
//     float *f = impulseTab;
//     for (i = 0; i < 256; i++) {
//         *f++ = d(engine);
//         *f++ = d(engine);
//         *f++ = d(engine);
//         *f++ = 1. - 2. * d(engine);
//     }
// }

// float catrom2(float d, int griddist) {
//     float x;
//     int i;
//     static float table[401];
//     static bool initialized = 0;
//     if (d >= griddist * griddist)
//         return 0;
//     if (!initialized) {
//         for (i = 0; i < 4 * 100 + 1; i++) {
//             x = i / (float)100;
//             x = sqrtf(x);
//             if (x < 1)
//                 table[i] = 0.5 * (2 + x * x * (-5 + x * 3));
//             else
//                 table[i] = 0.5 * (4 + x * (-8 + x * (5 - x)));
//         }
//         initialized = 1;
//     }
//     d = d * 100 + 0.5;
//     i = floor(d);
//     if (i >= 4 * 100 + 1)
//         return 0;
//     return table[i];
// }

// #define NEXT(h) (((h) + 1) & 255)

// float scnoise(float x, float y, float z, int pulsenum, int griddist) {
//     static int initialized;
//     float *fp = nullptr;
//     int i, j, k, h, n;
//     int ix, iy, iz;
//     float sum = 0;
//     float fx, fy, fz, dx, dy, dz, distsq;

//     /* Initialize the random impulse table if necessary. */
//     if (!initialized) {
//         impulseTabInit();
//         initialized = 1;
//     }
//     ix = floor(x);
//     fx = x - ix;
//     iy = floor(y);
//     fy = y - iy;
//     iz = floor(z);
//     fz = z - iz;

//     /* Perform the sparse convolution. */
//     for (i = -griddist; i <= griddist; i++) { //周围的grid ： 2*griddist+1
//         for (j = -griddist; j <= griddist; j++) {
//             for (k = -griddist; k <= griddist; k++) {         /* Compute voxel hash code. */
//                 h = INDEX(ix + i, iy + j, iz + k);            //PSN
//                 for (n = pulsenum; n > 0; n--, h = NEXT(h)) { /* Convolve filter and impulse. */
//                                                               //每个cell内随机产生pulsenum个impulse
//                     fp = &impulseTab[h * 4];                  // get impulse
//                     dx = fx - (i + *fp++);                    //i + *fp++   周围几个晶胞的脉冲
//                     dy = fy - (j + *fp++);
//                     dz = fz - (k + *fp++);
//                     distsq = dx * dx + dy * dy + dz * dz;
//                     sum += catrom2(distsq, griddist) *
//                            *fp; // 第四个fp 指向的就是每个点的权重    filter kernel在gabor noise里面变成了gabor kernel。
//                 }
//             }
//         }
//     }
//     return sum / pulsenum;
// }

// struct ImageNoise : INode {
//     virtual void apply() override {
//         auto image = std::make_shared<PrimitiveObject>();
//         auto griddist = get_input2<int>("griddist");
//         auto pulsenum = get_input2<int>("pulsenum");
//         auto size = get_input2<vec2i>("Size");
//         auto elementsize = get_input2<int>("elementsize");
//         image->verts.resize(size[0] * size[1]);
//         image->userData().set2("isImage", 1);
//         image->userData().set2("w", size[0]);
//         image->userData().set2("h", size[1]);

// //#pragma omp parallel
//         for (int i = 0; i < size[1]; i++) {
//             for (int j = 0; j < size[0]; j++) {
//                 i = i * 1/(elementsize);
//                 j = j * 1/(elementsize);
//                 //float x = (scnoise(i, 0, j, pulsenum, griddist) + 1) * 0.75;
//                 image->verts[i * size[0] + j][0] = (scnoise(i, 0, j, pulsenum, griddist) + 1) * 0.75;
//                 image->verts[i * size[0] + j][1] = (scnoise(j, 0, i, pulsenum, griddist) +1)*0.75;
//                 image->verts[i * size[0] + j][2] = (scnoise(0, i, j, pulsenum, griddist)+1)*0.75;
//             }
//         }

//         set_output("image", image);
        
//     }
// };

// ZENDEFNODE(ImageNoise, {
//     {
//         {"int", "pulsenum", "3"},
//         {"vec2i", "Size", "1024,1024"},
//         {"int", "elementsize", "50"},
//         {"int", "griddist", "2"}
//     },
//     {
//         {"image"},
//     },
//     {},
//     { "image" },
// });


struct ImageExtractColor : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto background = get_input2<std::string>("background");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        vec3f up = get_input2<vec3f>("high_threshold");
        vec3f low = get_input2<vec3f>("low_threshold");
        float upr = up[0]/255, upg = up[1]/255, upb = up[2]/255;
        float lr = low[0]/255,lg = low[1]/255,lb = low[2]/255;
        zeno::log_info("up:{}, {}, {}",upr,upg,upb);
        zeno::log_info("low:{}, {}, {}",lr,lg,lb);
        if(background == "transparent"){
            if(!image->has_attr("alpha")){
                image->verts.add_attr<float>("alpha");
                for(int i = 0; i < image->verts.size();i++){
                    image->verts.attr<float>("alpha")[i] = 1;
                }
            }
            for (auto i = 0; i < image->verts.size(); i++) {
                if(((upr < image->verts[i][0]) || (image->verts[i][0] < lr)) ||
                   ((upg < image->verts[i][1]) || (image->verts[i][1] < lg)) ||
                   ((upb < image->verts[i][2]) || (image->verts[i][2] < lb))){
                    image->verts.attr<float>("alpha")[i] = 0;
                }
            }
        }
        else if(background == "black"){
            for (auto i = 0; i < image->verts.size(); i++) {
                if(((upr < image->verts[i][0]) || (image->verts[i][0] < lr)) ||
                   ((upg < image->verts[i][1]) || (image->verts[i][1] < lg)) ||
                   ((upb < image->verts[i][2]) || (image->verts[i][2] < lb))){
                    image->verts[i] = {0,0,0};

                }
            }
        }
        else if(background == "white"){
            for (auto i = 0; i < image->verts.size(); i++) {
                if(((upr < image->verts[i][0]) || (image->verts[i][0] < lr)) ||
                   ((upg < image->verts[i][1]) || (image->verts[i][1] < lg)) ||
                   ((upb < image->verts[i][2]) || (image->verts[i][2] < lb))){
                    image->verts[i] = {1,1,1};
                }
            }
        }
        set_output("image", image);
    }
};
ZENDEFNODE(ImageExtractColor, {
    {
        {"image"},
        {"vec3f", "high_threshold", "255,255,255"},
        {"vec3f", "low_threshold", "0,0,0"},
        {"enum transparent black white", "background", "transparent"},
    },
    {
        {"image"},
    },
    {},
    { "image" },
});

struct ImageDelColor: INode {
    void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto background = get_input2<std::string>("background");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        vec3f up = get_input2<vec3f>("high_threshold");
        vec3f low = get_input2<vec3f>("low_threshold");
        float upr = up[0]/255, upg = up[1]/255, upb = up[2]/255;
        float lr = low[0]/255,lg = low[1]/255,lb = low[2]/255;
        if(background == "transparent"){
            if(!image->has_attr("alpha")){
                image->verts.add_attr<float>("alpha");
                for(int i = 0; i < image->verts.size();i++){
                    image->verts.attr<float>("alpha")[i] = 1;
                }
            }
            for (auto i = 0; i < image->verts.size(); i++) {
                if(((lr <= image->verts[i][0]) && (image->verts[i][0] <= upr)) &&
                   ((lg <= image->verts[i][1]) && (image->verts[i][1] <= upg)) &&
                   ((lb <= image->verts[i][2]) && (image->verts[i][2] <= upb))){
                    image->verts.attr<float>("alpha")[i] = 0;
                }
            }
        }
        else if(background == "black"){
            for (auto i = 0; i < image->verts.size(); i++) {
                if(((lr <= image->verts[i][0]) && (image->verts[i][0] <= upr)) &&
                   ((lg <= image->verts[i][1]) && (image->verts[i][1] <= upg)) &&
                   ((lb <= image->verts[i][2]) && (image->verts[i][2] <= upb))){
                    image->verts[i] = {0,0,0};
                }
            }
        }
        else if(background == "white"){
            for (auto i = 0; i < image->verts.size(); i++) {
                if(((lr <= image->verts[i][0]) && (image->verts[i][0] <= upr)) &&
                   ((lg <= image->verts[i][1]) && (image->verts[i][1] <= upg)) &&
                   ((lb <= image->verts[i][2]) && (image->verts[i][2] <= upb))){
                    image->verts[i] = {1,1,1};
                }
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageDelColor, {
    {
        {"image"},
        {"vec3f", "high_threshold", "255,255,255"},
        {"vec3f", "low_threshold", "0,0,0"},
        {"enum transparent black white", "background", "transparent"},
    },
    {
        {"image"},
    },
    {},
    {"image"},
});

struct ImageMatting: INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        auto imagemode = get_input2<std::string>("imagemode");
        auto maskmode = get_input2<std::string>("maskmode");

        if(imagemode == "origin"){
            if (!image->has_attr("alpha")) {
                image->verts.add_attr<float>("alpha");
                for (int i = 0; i < image->verts.size(); i++) {
                    image->verts.attr<float>("alpha")[i] = 1;
                }
            }
        }
        else if(imagemode == "deleteblack"){
            if (!image->has_attr("alpha")) {
                image->verts.add_attr<float>("alpha");
                for (int i = 0; i < image->verts.size(); i++) {
                    image->verts.attr<float>("alpha")[i] = 1;
                }
            }
            for (int i = 0; i < image->verts.size(); i++) {
                if(image->verts[i][0] <= 0.01 && image->verts[i][1] <= 0.01 &&
                   image->verts[i][2] <= 0.01 && image->verts.attr<float>("alpha")[i] != 0){
                    image->verts.attr<float>("alpha")[i] = 0;
                }
            }
        }
        else if(imagemode == "deletewhite"){
            if (!image->has_attr("alpha")) {
                image->verts.add_attr<float>("alpha");
                for (int i = 0; i < image->verts.size(); i++) {
                    image->verts.attr<float>("alpha")[i] = 1;
                }
            }
            for (int i = 0; i < image->verts.size(); i++) {
                if(image->verts[i][0] >= 0.99 && image->verts[i][1] >= 0.99 &&
                   image->verts[i][2] >= 0.99 && image->verts.attr<float>("alpha")[i] != 0){
                    image->verts.attr<float>("alpha")[i] = 0;
                }
            }
        }
        if (has_input("mask")) {
            auto gimage = get_input2<PrimitiveObject>("mask");
            auto &gud = gimage->userData();
            int wg = gud.get2<int>("w");
            int hg = gud.get2<int>("h");
            if (wg == w && hg == h) {
                if (maskmode == "black") {
                    if (gimage->verts.has_attr("alpha")) {
#pragma omp parallel for
                        for (int i = 0; i < h; i++) {
                            for (int j = 0; j < w; j++) {
                                if (gimage->verts.attr<float>("alpha")[i * w + j] != 0 &&
                                    image->verts.attr<float>("alpha")[i * w + j] != 0) {
                                    if (gimage->verts[i][0] <= 0.01 && gimage->verts[i][1] <= 0.01 &&
                                        gimage->verts[i][2] <= 0.01) {
                                        image->verts.attr<float>("alpha")[i * w + j] = 1;
                                    } else {
                                        image->verts.attr<float>("alpha")[i * w + j] = 0;
                                    }
                                } else {
                                    image->verts.attr<float>("alpha")[i * w + j] = 0;
                                }
                            }
                        }
                    } else {
#pragma omp parallel for
                        for (int i = 0; i < h; i++) {
                            for (int j = 0; j < w; j++) {
                                if (image->verts.attr<float>("alpha")[i * w + j] != 0) {
                                    if (gimage->verts[i * w + j][0] == 0 && gimage->verts[i * w + j][1] == 0 &&
                                        gimage->verts[i * w + j][2] == 0) {
                                        image->verts.attr<float>("alpha")[i * w + j] = 1;
                                    } else {
                                        image->verts.attr<float>("alpha")[i * w + j] = 0;
                                    }
                                }
                            }
                        }
                    }
                } else if (maskmode == "white") {
                    if (gimage->verts.has_attr("alpha")) {
#pragma omp parallel for
                        for (int i = 0; i < h; i++) {
                            for (int j = 0; j < w; j++) {
                                if (gimage->verts.attr<float>("alpha")[i * w + j] != 0 &&
                                    image->verts.attr<float>("alpha")[i * w + j] != 0) {
                                    if (gimage->verts[i][0] >= 0.99 && gimage->verts[i][1] >= 0.99 &&
                                        gimage->verts[i][2] >= 0.99) {
                                        image->verts.attr<float>("alpha")[i * w + j] = 1;
                                    } else {
                                        image->verts.attr<float>("alpha")[i * w + j] = 0;
                                    }
                                } else {
                                    image->verts.attr<float>("alpha")[i * w + j] = 0;
                                }
                            }
                        }
                    } else {
#pragma omp parallel for
                        for (int i = 0; i < h; i++) {
                            for (int j = 0; j < w; j++) {
                                if (image->verts.attr<float>("alpha")[i * w + j] != 0) {
                                    if (gimage->verts[i * w + j][0] == 0 && gimage->verts[i * w + j][1] == 0 &&
                                        gimage->verts[i * w + j][2] == 0) {
                                        image->verts.attr<float>("alpha")[i * w + j] = 1;
                                    } else {
                                        image->verts.attr<float>("alpha")[i * w + j] = 0;
                                    }
                                }
                            }
                        }
                    }
                } else if (maskmode == "gray_black") {
                    if (gimage->verts.has_attr("alpha")) {
#pragma omp parallel for
                        for (int i = 0; i < h; i++) {
                            for (int j = 0; j < w; j++) {
                                if (gimage->verts.attr<float>("alpha")[i * w + j] != 0 &&
                                    image->verts.attr<float>("alpha")[i * w + j] != 0) {
                                    image->verts.attr<float>("alpha")[i * w + j] = 1 - gimage->verts[i * w + j][0];
                                } else {
                                    image->verts.attr<float>("alpha")[i * w + j] = 0;
                                }
                            }
                        }
                    } else {
#pragma omp parallel for
                        for (int i = 0; i < h; i++) {
                            for (int j = 0; j < w; j++) {
                                if (image->verts.attr<float>("alpha")[i * w + j] != 0) {
                                    image->verts.attr<float>("alpha")[i * w + j] = 1 - gimage->verts[i * w + j][0];
                                }
                            }
                        }
                    }
                } else if (maskmode == "gray_white") {
                    if (gimage->verts.has_attr("alpha")) {
#pragma omp parallel for
                        for (int i = 0; i < h; i++) {
                            for (int j = 0; j < w; j++) {
                                if (gimage->verts.attr<float>("alpha")[i * w + j] != 0 &&
                                    image->verts.attr<float>("alpha")[i * w + j] != 0) {
                                    image->verts.attr<float>("alpha")[i * w + j] = gimage->verts[i * w + j][0];
                                } else {
                                    image->verts.attr<float>("alpha")[i * w + j] = 0;
                                }
                            }
                        }
                    } else {
#pragma omp parallel for
                        for (int i = 0; i < h; i++) {
                            for (int j = 0; j < w; j++) {
                                if (image->verts.attr<float>("alpha")[i * w + j] != 0) {
                                    image->verts.attr<float>("alpha")[i * w + j] = gimage->verts[i * w + j][0];
                                }
                            }
                        }
                    }
                }
                else if (maskmode == "alpha") {
                    if (gimage->verts.has_attr("alpha")) {
                        image->verts.attr<float>("alpha") = gimage->verts.attr<float>("alpha");
                    } else {
#pragma omp parallel for
                        for (int i = 0; i < h; i++) {
                            for (int j = 0; j < w; j++) {
                                if (image->verts.attr<float>("alpha")[i * w + j] != 0) {
                                    image->verts.attr<float>("alpha")[i * w + j] = 1;
                                }
                            }
                        }
                    }
                }
            }
            //todo
            else if (wg < w && hg < h) {
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageMatting, {
    {
        {"image"},
        {"mask"},
        {"enum origin deleteblack deletewhite", "imagemode", "origin"},
        {"enum gray_black gray_white black white alpha", "maskmode", "gray_black"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
});

//根据灰度进行上色
struct MaskEdit: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input<PrimitiveObject>("image");
        UserData &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        for(int i = 0;i < image->size();i++){

        }

        set_output("image", image);
    }
};
ZENDEFNODE(MaskEdit, {
    {
        {"image"},
        {"int", "rows", "2"},
        {"int", "cols", "2"},
    },
    {
        {"image"},
    },
    {},
    {"deprecated"},
});


struct ImageLevels: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input<PrimitiveObject>("image");
        auto inputLevels = get_input2<vec2f>("Input Levels");
        auto outputLevels = get_input2<vec2f>("Output Levels");
        auto gamma = get_input2<float>("gamma");//range  0.01 - 9.99
        auto channel = get_input2<std::string>("channel");
        auto clamp = get_input2<bool>("Clamp Output");
        auto autolevel = get_input2<bool>("Auto Level");
        UserData &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        float inputRange = inputLevels[1] - inputLevels[0];
        float outputRange = outputLevels[1] - outputLevels[0];
        float inputMin = inputLevels[0];
        float outputMin = outputLevels[0];
        float gammaCorrection = 1.0f / gamma;
        float MinBlue, MaxBlue, MinRed, MaxRed, MinGreen, MaxGreen = 0.0f;
        //calculate histogram
        if (autolevel) {
            std::vector<int> histogramred(256, 0);
            std::vector<int> histogramgreen(256, 0);
            std::vector<int> histogramblue(256, 0);
            for (int i = 0; i < w * h; i++) {
                histogramred[zeno::clamp(int(image->verts[i][0] * 255.99), 0, 255)]++;//int 没问题吗
                histogramgreen[zeno::clamp(int(image->verts[i][1] * 255.99), 0, 255)]++;
                histogramblue[zeno::clamp(int(image->verts[i][2] * 255.99), 0, 255)]++;
            }
            int total = w * h;
            int sum = 0;
            //Red channel
            for (int i = 0; i < 256; i++) {
                sum += histogramred[i];
                if (sum  >= total * 0.001f) {
                    MinRed = i;
                    break;
                }
            }
            sum = 0;
            for (int i = 255; i >= 0; i--) {
                sum += histogramred[i];
                if (sum  >= total * 0.001f) {
                    MaxRed = i;
                    break;
                }
            }
            //Green channel
            sum = 0;
            for (int i = 0; i < 256; i++) {
                sum += histogramgreen[i];
                if (sum  >= total * 0.001f) {
                    MinGreen = i;
                    break;
                }
            }
            sum = 0;
            for (int i = 255; i >= 0; i--) {
                sum += histogramgreen[i];
                if (sum  >= total * 0.001f) {
                    MaxGreen = i;
                    break;
                }
            }
            //Blue channel
            sum = 0;
            for (int i = 0; i < 256; i++) {
                sum += histogramblue[i];
                if (sum  >= total * 0.001f) {
                    MinBlue = i;
                    break;
                }
            }
            sum = 0;
            for (int i = 255; i >= 0; i--) {
                sum += histogramblue[i];
                if (sum  >= total * 0.001f) {
                    MaxBlue = i;
                    break;
                }
            }
            //inputMin = std::min(std::min(MinRed, MinGreen), MinBlue) / 255.0f;//auto contrast?  对于灰度图像，由于只有一个通道，自动对比度和自动色阶实际上算法相同？
            //inputRange = (std::max(std::max(MaxRed, MaxGreen), MaxBlue) - inputMin) / 255.0f;
            /*// 根据计算的值影响level参数
            inputMin = min / 255.0f;
            inputRange = (max - min) / 255.0f;*/
        }
        MinRed /= 255.0f, MinGreen /= 255.0f, MinBlue /= 255.0f, MaxRed /= 255.0f, MaxGreen /= 255.0f, MaxBlue /= 255.0f;

        if(autolevel){
#pragma omp parallel for
            for (int i = 0; i < w * h; i++) {
                vec3f &v = image->verts[i];
                v[0] = (v[0] < MinRed) ? MinRed : v[0];
                v[1] = (v[1] < MinGreen) ? MinGreen : v[1];
                v[2] = (v[2] < MinBlue) ? MinBlue : v[2];
                v[0] = (v[0] - MinRed) / (MaxRed - MinRed);
                v[1] = (v[1] - MinGreen) / (MaxGreen - MinGreen);
                v[2] = (v[2] - MinBlue) / (MaxBlue - MinBlue);
                v = clamp ? zeno::clamp((v * outputRange + outputMin), 0, 1) : (v * outputRange + outputMin);
            }
        }
        else if (channel == "All") {
#pragma omp parallel for
            for (int i = 0; i < w * h; i++) {
                vec3f &v = image->verts[i];
                v[0] = (v[0] < inputMin) ? inputMin : v[0];
                v[1] = (v[1] < inputMin) ? inputMin : v[1];
                v[2] = (v[2] < inputMin) ? inputMin : v[2];
                v = (v - inputMin) / inputRange; 
                v = pow(v, gammaCorrection);
                v = clamp ? zeno::clamp((v * outputRange + outputMin), 0, 1) : (v * outputRange + outputMin);
            }
            if(image->has_attr("alpha")){
                auto &alphaAttr = image->verts.attr<float>("alpha");
#pragma omp parallel for
                for (int i = 0; i < w * h; i++) {
                    alphaAttr[i] = (alphaAttr[i] < inputMin) ? inputMin : alphaAttr[i];
                    alphaAttr[i] = (alphaAttr[i] - inputMin) / inputRange;
                    alphaAttr[i] = pow(alphaAttr[i], gammaCorrection) * outputRange + outputMin;
                    alphaAttr[i] = clamp ? zeno::clamp(alphaAttr[i], 0, 1) : alphaAttr[i];
                }
            }
        }
        else if (channel == "R") {
#pragma omp parallel for
        for (int i = 0; i < w * h; i++) {
                float &v = image->verts[i][0];
                if (v < inputMin) v = inputMin;
                v = (v - inputMin) / inputRange;
                v = pow(v, gammaCorrection);
                v = clamp ? zeno::clamp((v * outputRange + outputMin), 0, 1) : (v * outputRange + outputMin);
            }
        }

        else if (channel == "G") {
#pragma omp parallel for
        for (int i = 0; i < w * h; i++) {
                float &v = image->verts[i][1];
                if (v < inputMin) v = inputMin;
                v = (v - inputMin) / inputRange;
                v = pow(v, gammaCorrection);
                v = clamp ? zeno::clamp((v * outputRange + outputMin), 0, 1) : (v * outputRange + outputMin);
            }
        }
        
        else if (channel == "B") {
#pragma omp parallel for
        for (int i = 0; i < w * h; i++) {
                float &v = image->verts[i][2];
                if (v < inputMin) v = inputMin;
                v = (v - inputMin) / inputRange;
                v = pow(v, gammaCorrection);
                v = clamp ? zeno::clamp((v * outputRange + outputMin), 0, 1) : (v * outputRange + outputMin);
            }
        }
        
        else if (channel == "A") {
        if(image->has_attr("alpha")){
            auto &alphaAttr = image->verts.attr<float>("alpha");
#pragma omp parallel for
            for (int i = 0; i < w * h; i++) {
                alphaAttr[i] = (alphaAttr[i] < inputMin) ? inputMin : alphaAttr[i];
                alphaAttr[i] = (alphaAttr[i] - inputMin) / inputRange;
                alphaAttr[i] = pow(alphaAttr[i], gammaCorrection) * outputRange + outputMin;
                alphaAttr[i] = clamp ? zeno::clamp(alphaAttr[i], 0, 1) : alphaAttr[i];
                }
            }
            else{
                zeno::log_error("no alpha channel");
            }
        }

        set_output("image", image);
    }
};
ZENDEFNODE(ImageLevels, {
    {
        {"image"},
        {"vec2f", "Input Levels", "0, 1"},
        {"float", "gamma", "1"},
        {"vec2f", "Output Levels", "0, 1"},
        {"enum All R G B A", "channel", "RGB"},
        {"bool", "Auto Level", "0"},
        {"bool", "Clamp Output", "1"},
    },
    {
        {"image"},
    },
    {},
    {"image"},
});
}
}