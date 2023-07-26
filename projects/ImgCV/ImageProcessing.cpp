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

struct ImageBlur : INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto xsize = get_input2<int>("xsize");
        auto ysize = get_input2<int>("ysize");
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
    { "image" },
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
        auto color = get_input2<vec3f>("Color");
        auto size = get_input2<vec2i>("Size");
        image->verts.resize(size[0] * size[1]);
        image->userData().set2("isImage", 1);
        image->userData().set2("w", size[0]);
        image->userData().set2("h", size[1]);

#pragma omp parallel
        for (int i = 0; i < size[1]; i++) {
            for (int j = 0; j < size[0]; j++) {
                image->verts[i * size[0] + j] = {color[0], color[1], color[2]};
            }
        }

        set_output("image", image);
        
    }
};

ZENDEFNODE(ImageColor, {
    {
        {"vec3f", "Color", "1,1,1"},
        {"vec2i", "Size", "1024,1024"},
    },
    {
        {"image"},
    },
    {},
    { "image" },
});


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

struct ImageDelAlpha: INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        if(image->verts.has_attr("alpha")){
            image->verts.erase_attr("alpha");
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageDelAlpha, {
    {
        {"image"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
});

struct ImageAddAlpha: INode {
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        auto maskmode = get_input2<std::string>("maskmode");
        image->verts.add_attr<float>("alpha");
        for(int i = 0;i < image->size();i++){
            image->verts.attr<float>("alpha")[i] = 1;
        }
        if (has_input("mask")) {
            auto gimage = get_input2<PrimitiveObject>("mask");
            auto &gud = gimage->userData();
            int wg = gud.get2<int>("w");
            int hg = gud.get2<int>("h");
            if (wg == w && hg == h) {
                 if (maskmode == "gray_black") {
                    if (gimage->verts.has_attr("alpha")) {
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
        }
        set_output("image", image);
    }
};
ZENDEFNODE(ImageAddAlpha, {
    {
        {"image"},
        {"mask"},
        {"enum alpha gray_black gray_white", "maskmode", "alpha"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
});

struct ImageCut: INode {
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
        set_output("image", image2);
    }
};
ZENDEFNODE(ImageCut, {
    {
        {"image"},
        {"enum normal mirror", "tilemode", "normal"},
        {"int", "rows", "2"},
        {"int", "cols", "2"},
    },
    {
        {"deprecated"},
    },
    {},
    {"image"},
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

struct ImageShape: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input<PrimitiveObject>("image");
        auto rows = get_input2<int>("rows");
        auto cols = get_input2<int>("cols");
        UserData &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        set_output("image", image);
    }
};
ZENDEFNODE(ImageShape, {
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
        UserData &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        float inputRange = inputLevels[1] - inputLevels[0];
        float outputRange = outputLevels[1] - outputLevels[0];
        float inputMin = inputLevels[0];
        float outputMin = outputLevels[0];
        float gammaCorrection = 1.0f / gamma;

        if (channel == "RGB") {
#pragma omp parallel for
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                vec3f &v = image->verts[i * w + j];
                v[0] = (v[0] < inputMin) ? inputMin : v[0];
                v[1] = (v[1] < inputMin) ? inputMin : v[1];
                v[2] = (v[2] < inputMin) ? inputMin : v[2];
                v = (v - inputMin) / inputRange; 
                v = pow(v, gammaCorrection);
                v = v * outputRange + outputMin;
            }
        }
        }

        else if (channel == "R") {
#pragma omp parallel for
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) { 
                float &v = image->verts[i * w + j][0];
                if (v < inputMin) v = inputMin;
                v = (v - inputMin) / inputRange;
                v = pow(v, gammaCorrection);
                v = v * outputRange + outputMin;
            }
        }
        }

        else if (channel == "G") {
#pragma omp parallel for
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) { 
                float &v = image->verts[i * w + j][1];
                if (v < inputMin) v = inputMin;
                v = (v - inputMin) / inputRange;
                v = pow(v, gammaCorrection);
                v = v * outputRange + outputMin;
            }
        }
        }
        
        else if (channel == "B") {
#pragma omp parallel for
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) { 
                float &v = image->verts[i * w + j][2];
                if (v < inputMin) v = inputMin;
                v = (v - inputMin) / inputRange;
                v = pow(v, gammaCorrection);
                v = v * outputRange + outputMin;
            }
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
        //{"bool", "auto level", "false"}, //auto level
        {"enum RGB R G B", "channel", "RGB"},
    },
    {
        {"image"},
    },
    {},
    {"image"},
});
}
}