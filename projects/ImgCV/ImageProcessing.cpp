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
#include <zeno/utils/image_proc.h>
#include <cmath>
#include <zeno/utils/log.h>
#include <opencv2/opencv.hpp>

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

static vec3f RGBtoXYZ(vec3f rgb) {//INPUT RANGE 0-1
    float r = rgb[0];
    float g = rgb[1];
    float b = rgb[2];
    if (r > 0.04045) {
        r = pow((r + 0.055) / 1.055, 2.4);
    } else {
        r = r / 12.92;
    }
    if (g > 0.04045) {
        g = pow((g + 0.055) / 1.055, 2.4);
    } else {
        g = g / 12.92;
    }
    if (b > 0.04045) {
        b = pow((b + 0.055) / 1.055, 2.4);
    } else {
        b = b / 12.92;
    }
    r *= 100;
    g *= 100;
    b *= 100;
    return vec3f(0.412453 * r + 0.357580 * g + 0.180423 * b, 0.212671 * r + 0.715160 * g + 0.072169 * b, 0.019334 * r + 0.119193 * g + 0.950227 * b);
}

static vec3f XYZtoRGB(vec3f xyz) {//OUTPUT RANGE 0-1
    float x = xyz[0];
    float y = xyz[1];
    float z = xyz[2];
    x /= 100;
    y /= 100;
    z /= 100;
    float r = x * 3.2406 + y * -1.5372 + z * -0.4986;
    float g = x * -0.9689 + y * 1.8758 + z * 0.0415;
    float b = x * 0.0557 + y * -0.2040 + z * 1.0570;
    if (r > 0.0031308) {
        r = 1.055 * pow(r, 1 / 2.4) - 0.055;
    } else {
        r = 12.92 * r;
    }
    if (g > 0.0031308) {
        g = 1.055 * pow(g, 1 / 2.4) - 0.055;
    } else {
        g = 12.92 * g;
    }
    if (b > 0.0031308) {
        b = 1.055 * pow(b, 1 / 2.4) - 0.055;
    } else {
        b = 12.92 * b;
    }
    return vec3f(r, g, b);
}

static vec3f XYZtoLab(vec3f xyz) {
    float x = xyz[0];
    float y = xyz[1];
    float z = xyz[2];
    x /= 95.047;
    y /= 100;
    z /= 108.883;
    if (x > 0.008856) {
        x = pow(x, 1.0 / 3.0);
    } else {
        x = (7.787 * x) + (16.0 / 116.0);
    }
    if (y > 0.008856) {
        y = pow(y, 1.0 / 3.0);
    } else {
        y = (7.787 * y) + (16.0 / 116.0);
    }
    if (z > 0.008856) {
        z = pow(z, 1.0 / 3.0);
    } else {
        z = (7.787 * z) + (16.0 / 116.0);
    }
    return vec3f((116 * y) - 16, 500 * (x - y), 200 * (y - z));
}

static vec3f LabtoXYZ(vec3f lab) {
    float l = lab[0];
    float a = lab[1];
    float b = lab[2];
    float y = (l + 16) / 116;
    float x = a / 500 + y;
    float z = y - b / 200;
    if (pow(y, 3) > 0.008856) {
        y = pow(y, 3);
    } else {
        y = (y - 16.0 / 116.0) / 7.787;
    }
    if (pow(x, 3) > 0.008856) {
        x = pow(x, 3);
    } else {
        x = (x - 16.0 / 116.0) / 7.787;
    }
    if (pow(z, 3) > 0.008856) {
        z = pow(z, 3);
    } else {
        z = (z - 16.0 / 116.0) / 7.787;
    }
    return vec3f(x * 95.047, y * 100, z * 108.883);
}

static vec3f RGBtoLab(vec3f rgb) {//input range 0-1
    return XYZtoLab(RGBtoXYZ(rgb));
}

static vec3f LabtoRGB(vec3f lab) {//output range 0-1
    return XYZtoRGB(LabtoXYZ(lab));
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


/*struct ImageResize: INode {//TODO::FIX BUG
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

        float scaleX = static_cast<float>(w) / width;
        float scaleY = static_cast<float>(h) / height;

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
});*/

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
struct ImageRotate: INode {//TODO::transform and rorate
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

struct ImageFlip: INode {
    void apply() override {
        auto fliphori = get_input2<bool>("Flip Horizontally");
        auto flipvert = get_input2<bool>("Flip Vertically");
        auto image = get_input<PrimitiveObject>("image");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        if(fliphori&&flipvert){
            image_flip_horizontal(image->verts.data(), w, h);
            image_flip_vertical(image->verts.data(), w, h);
            if (image->verts.has_attr("alpha")) {
                auto alpha = image->verts.attr<float>("alpha");
                image_flip_horizontal(alpha.data(), w, h);
                image_flip_vertical(alpha.data(), w, h);
            }
        }
        else if(fliphori&&!flipvert){
            image_flip_horizontal(image->verts.data(), w, h);
            if (image->verts.has_attr("alpha")) {
                auto alpha = image->verts.attr<float>("alpha");
                image_flip_horizontal(alpha.data(), w, h);
            }
        }
        else if(!fliphori&&flipvert){
            image_flip_vertical(image->verts.data(), w, h);
            if (image->verts.has_attr("alpha")) {
                auto alpha = image->verts.attr<float>("alpha");
                image_flip_vertical(alpha.data(), w, h);
            }
        }
        else if(!fliphori&&!flipvert){
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageFlip, {
    {
        {"image"},
        {"bool", "Flip Horizontally", "0"},
        {"bool", "Flip Vertically", "0"},
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

struct ImageEditHSV : INode {//TODO::FIX BUG
    virtual void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        float H = 0, S = 0, V = 0;
        float Hi = get_input2<float>("H");
        float Si = get_input2<float>("S");
        float Vi = get_input2<float>("V");
        for (auto i = 0; i < image->verts.size(); i++) {
            float R = image->verts[i][0];
            float G = image->verts[i][1];
            float B = image->verts[i][2];
            zeno::RGBtoHSV(R, G, B, H, S, V);
            //S = S + (S - 0.5)*(Si-1);
            //V = V + (V - 0.5)*(Vi-1);
            //S = S + (Si - 1) * (S < 0.5 ? S : 1.0 - S);
            //V = V + (Vi - 1) * (V < 0.5 ? V : 1.0 - V);
            H = fmod(H + Hi, 360.0);
            S = S * Si;
            V = V * Vi;
            zeno::HSVtoRGB(H, S, V, R, G, B);
            image->verts[i][0] = R;
            image->verts[i][1] = G;
            image->verts[i][2] = B;
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageEditHSV, {
    {
        {"image"},
        {"float", "H", "0"},
        {"float", "S", "1"},
        {"float", "V", "1"},
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

//                cv::stackBlur(imagecvin,imagecvout,cv::Size(kernelSize, kernelSize));

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
        {"enum Gaussian Box Median Bilateral Stack", "type", "Gaussian"},
        {"int", "kernelSize", "5"},
        {"float", "GaussianSigma", "3"},//fast gaussian only effect by sigma  等参数分开显示再移开
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

/* 将高度图像转换为法线贴图 */
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
                if (i == 0 || i == h || j == 0 || j == w) {
                    normalmap->verts[i * w + j] = {0, 0, 1};
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
            vec3f &v = image->verts[i];
            if(mode=="Average"){
                float avg = (v[0] + v[1] + v[2]) / 3;
                v = vec3f(avg);
            }
            else if(mode=="Luminance"){
                float lumi = 0.3 * v[0] + 0.59 * v[1] + 0.11 * v[2];//(GIMP/PS)
                v =vec3f(lumi);
            }
            else if(mode=="Red"){
                v = vec3f(v[0]);
            }
            else if(mode=="Green"){
                v = vec3f(v[1]);
            }
            else if(mode=="Blue"){
                v = vec3f(v[2]);
            }
            else if(mode=="MaxComponent"){
                float max = std::max(v[0], std::max(v[1], v[2]));
                v = vec3f(max);
            }
            else if(mode=="MinComponent"){
                float min = std::min(v[0], std::min(v[1], v[2]));
                v = vec3f(min);
            }
        }
        set_output("image", image);
    }
};
ZENDEFNODE(ImageGray, {
    {
        {"image"},
        {"enum Average Luminance Red Green Blue MaxComponent MinComponent", "mode", "Average"},
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
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                vec3f rgb = image->verts[i * w + j];
                imagecvin.at<cv::Vec3f>(i, j) = {rgb[0], rgb[1], rgb[2]};
            }
        }

        cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(kheight, kwidth));
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
                image->verts[i] = {color[0], color[1], color[2]};
                alphaAttr[i] = color[3];
            }
        }
        else{
            for (int i = 0; i < vertsize ; i++) {
                image->verts[i] = {color[0], color[1], color[2]};
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
                image->verts[i] = {color[0], color[1], color[2]};
                alphaAttr[i] = alpha;
            }
        }
        else{
            for (int i = 0; i < vertsize ; i++) {
                image->verts[i] = {color[0], color[1], color[2]};
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

struct ImageClamp: INode {//Add Unpremultiplied Space Option?
    void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto background = get_input2<std::string>("ClampedValue");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        auto up = get_input2<float>("Max");
        auto low = get_input2<float>("Min");
        if(background == "LimitValue"){
        for (auto i = 0; i < image->verts.size(); i++) {
            image->verts[i] = zeno::clamp(image->verts[i], low, up);
            }
        }
        else if(background == "Black"){
            for (auto i = 0; i < image->verts.size(); i++) {
                vec3f &v = image->verts[i];
                for(int j = 0; j < 3; j++){
                    if((v[j]<low) || (v[j]>up)){
                        v[j] = 0;
                    }
                }
            }
        }
        else if(background == "White"){
            for (auto i = 0; i < image->verts.size(); i++) {
                vec3f &v = image->verts[i];
                for(int j = 0; j < 3; j++){
                    if((v[j]<low) || (v[j]>up)){
                        v[j] = 1;
                    }
                }
            }
        }

        set_output("image", image);
    }
};

ZENDEFNODE(ImageClamp, {
    {
        {"image"},
        {"float", "Max", "1"},
        {"float", "Min", "0"},
        {"enum LimitValue Black White", "ClampedValue", "LimitValue"},
    },
    {
        {"image"},
    },
    {},
    {"image"},
});
/*
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
});*/

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
                histogramred[zeno::clamp(int(image->verts[i][0] * 255.99), 0, 255)]++;
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
        {"enum All R G B A", "channel", "All"},
        {"bool", "Auto Level", "0"},
        {"bool", "Clamp Output", "1"},
    },
    {
        {"image"},
    },
    {},
    {"image"},
});

struct ImageQuantization: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input<PrimitiveObject>("image");
        int clusternumber = get_input2<int>("Number of Color");
        bool outputcenter = get_input2<bool>("Output Cluster Centers");
        //int simplifyscale = get_input2<int>("Image Simplification Scale");
        auto clusterattribute = get_input2<std::string>("Cluster Attribute");
        UserData &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        auto &imagepos = image->verts.attr<vec3f>("pos");
        const int HistogramSize = 64 * 64 * 64;// (256/simplifyscale)  256 / 4 = 64
        std::vector<vec3f> labs(HistogramSize, vec3f(0.0f));
        std::vector<vec3f> seeds(clusternumber, vec3f(0.0f));
        std::vector<vec3f> newseeds(clusternumber, vec3f(0.0f));
        std::vector<int> weights(HistogramSize, 0);
        std::vector<int> seedsweight(clusternumber, 0);
        std::vector<int> clusterindices(HistogramSize, 0);
        #pragma omp parallel
        {
            std::vector<vec3f> local_labs(HistogramSize, vec3f(0.0f));
            std::vector<int> local_weights(HistogramSize, 0);

            #pragma omp for
            for (int i = 0; i < w * h; i++) {
                int index = ((zeno::clamp(int(imagepos[i][0] * 255.99), 0, 255) / 4) * 64
                + (zeno::clamp(int(imagepos[i][1] * 255.99), 0, 255) / 4)) * 64
                + (zeno::clamp(int(imagepos[i][2] * 255.99), 0, 255) / 4);

                local_labs[index] += RGBtoLab(imagepos[i]);
                local_weights[index]++;
            }

            #pragma omp critical
            {
                for (int i = 0; i < HistogramSize; i++) {
                    labs[i] += local_labs[i];
                    weights[i] += local_weights[i];
                }
            }
        }
        int maxindex = 0;
        vec3f seedcolor;
        const int squaredSeparationCoefficient = 1400;// For photos, we can use a higher coefficient, from 900 to 6400
        auto weightclone = weights;
        for (int i = 0; i < clusternumber; i++) {
            maxindex = std::max_element(weightclone.begin(), weightclone.end()) - weightclone.begin();
            if(weightclone[maxindex] == 0){
                break;
            }
            seedcolor = labs[maxindex] / weights[maxindex];
            seeds[i] = seedcolor;
            weightclone[maxindex] = 0;
            for (int i = 0; i < HistogramSize; i++)
            {
                if(weightclone[i] > 0 ) {
                    weightclone[i] *= (1 - exp(-zeno::lengthSquared(seedcolor - labs[i] / weights[i]) / squaredSeparationCoefficient));
                    }
                }
            }
        bool optimumreached = false;
        while(!optimumreached){
            optimumreached = true;
            std::fill(newseeds.begin(), newseeds.end(), vec3f(0.0f));
            std::fill(seedsweight.begin(), seedsweight.end(), 0);
            #pragma omp parallel
            {
                std::vector<vec3f> local_newseeds(clusternumber, vec3f(0.0f));
                std::vector<int> local_seedsweight(clusternumber, 0);

                #pragma omp for
                for (int i = 0; i < HistogramSize; i++) {
                    if(weights[i] == 0){
                        continue;
                    }
                    //get closest seed index
                    int mindist = 100000000;
                    int clusterindex;
                    for(int j = 0; j < clusternumber; j++){
                        auto dist = zeno::lengthSquared(labs[i] / weights[i] - seeds[j]);
                        if(dist < mindist){
                            mindist = dist;
                            clusterindex = j;
                        }
                    }
                    if(clusterindices[i] != clusterindex && optimumreached){
                        optimumreached = false;
                    }
                    clusterindices[i] = clusterindex;
                    // Accumulate colors and weights per cluster.
                    local_newseeds[clusterindex] += labs[i];
                    local_seedsweight[clusterindex] += weights[i];
                }

                #pragma omp critical
                {
                    for (int i = 0; i < clusternumber; i++) {
                        newseeds[i] += local_newseeds[i];
                        seedsweight[i] += local_seedsweight[i];
                    }
                }
            }
            // Average accumulated colors to get new seeds.
            for (int i = 0; i < clusternumber; i++) {// update seeds
                if (seedsweight[i] == 0){
                    seeds[i] = vec3f(0.0f);
                }
                else{
                    seeds[i] = newseeds[i] / seedsweight[i];
                }
            }
        }
        auto &clusterattr = image->verts.add_attr<int>(clusterattribute);
        //export pallete
        if (outputcenter) {
            image->verts.resize(clusternumber);
            image->verts.update();
            image->userData().set2("w", clusternumber);
            image->userData().set2("h", 1);
            for (int i = 0; i < clusternumber; i++) {
                image->verts[i] = LabtoRGB(seeds[i]);
                clusterattr[i] = i;
                }
        }
        else{
#pragma omp parallel for
            for (int i = 0; i < w * h; i++) {
                int index = ((zeno::clamp(int(imagepos[i][0] * 255.99), 0, 255) / 4) * 64 + 
                            (zeno::clamp(int(imagepos[i][1] * 255.99), 0, 255) / 4)) * 64 + 
                            (zeno::clamp(int(imagepos[i][2] * 255.99), 0, 255) / 4);
                image->verts[i] = LabtoRGB(seeds[clusterindices[index]]);
                clusterattr[i] = clusterindices[index];
            }
        }
        set_output("image", image);
    }
};
ZENDEFNODE(ImageQuantization, {
    {
        {"image"},
        {"int", "Number of Color", "5"},
        {"bool", "Output Cluster Centers", "1"},
        {"string", "Cluster Attribute", "cluster"},
    },
    {
        {"image"},
    },
    {},
    {"image"},
});

}
}