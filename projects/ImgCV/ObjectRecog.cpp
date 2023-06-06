#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/scope_exit.h>
#include <cmath>
#include <zeno/utils/log.h>
#include <filesystem>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/xfeatures2d.hpp>

using namespace cv;

namespace zeno {

namespace {

static void zenoedge(std::shared_ptr<PrimitiveObject> & grayImage, int width, int height, std::vector<float>& dx, std::vector<float>& dy)
{
    dx.resize(width * height);
    dy.resize(width * height);
#pragma omp parallel for
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
    zenoedge(grayImage, width, height, dx, dy);
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

struct EdgeDetect : INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        int threshold = get_input2<int>("threshold");
        int maxThreshold = get_input2<int>("maxThreshold");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        if(mode=="zeno_gray"){
            std::vector<float> dx,dy;
            zenoedge(image, w, h, dx, dy);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {

                    float gradient = std::sqrt(pow(dx[i * w + j],2) + pow(dy[i * w + j],2));
                    image->verts[i * w + j] = {gradient,gradient,gradient};
                }
            }
            set_output("image", image);
        }

        if(mode=="zeno_threshold"){
            std::vector<float> dx,dy;
            zenoedge(image, w, h, dx, dy);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {

                    float gradient = std::sqrt(pow(dx[i * w + j],2) + pow(dy[i * w + j],2));

                    if (gradient * 255 > threshold) {
                        image->verts[i * w + j] = {1,1,1};
                    }
                    else {
                        image->verts[i * w + j] = {0,0,0};
                    }
                }
            }
            set_output("image", image);
        }
        if(mode=="sobel_gray"){
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

            cv::Mat gradX, gradY;
            cv::Sobel(imagecvin, gradX, CV_32F, 1, 0);
            cv::Sobel(imagecvin, gradY, CV_32F, 0, 1);

            cv::Mat gradientMagnitude, gradientDirection;
            cv::cartToPolar(gradX, gradY, imagecvout, gradientDirection, true);

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
        if(mode=="sobel_threshold"){
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

            cv::Mat gradX, gradY;
            cv::Sobel(imagecvin, gradX, CV_32F, 1, 0);
            cv::Sobel(imagecvin, gradY, CV_32F, 0, 1);

            cv::Mat gradientMagnitude, gradientDirection;
            cv::cartToPolar(gradX, gradY, gradientMagnitude, gradientDirection, true);

            // 应用阈值
            cv::threshold(gradientMagnitude, imagecvout, threshold, maxThreshold, cv::THRESH_BINARY);
            for (int i = 0; i < h; i++) {

                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
        if(mode=="roberts_gray"){
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat robertsX, robertsY;
            cv::Mat magnitude;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2])/3;
                    imagecvin.at<float>(i, j) = var;
                }
            }
            cv::Mat kernelX = (cv::Mat_<float>(2, 2) << 1, 0, 0, -1);
            cv::filter2D(imagecvin, robertsX, -1, kernelX);

            cv::Mat kernelY = (cv::Mat_<float>(2, 2) << 0, 1, -1, 0);
            cv::filter2D(imagecvin, robertsY, -1, kernelY);

            cv::magnitude(robertsX, robertsY, imagecvout);

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }

        if(mode=="roberts_threshold"){
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat robertsX, robertsY;
            cv::Mat magnitude;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2])/3;
                    imagecvin.at<float>(i, j) = var;
                }
            }
            cv::Mat kernelX = (cv::Mat_<float>(2, 2) << 1, 0, 0, -1);
            cv::filter2D(imagecvin, robertsX, -1, kernelX);
            cv::Mat kernelY = (cv::Mat_<float>(2, 2) << 0, 1, -1, 0);
            cv::filter2D(imagecvin, robertsY, -1, kernelY);
            cv::magnitude(robertsX, robertsY, magnitude);
            cv::threshold(magnitude, imagecvout, threshold, maxThreshold, cv::THRESH_BINARY);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
        if(mode=="roberts_gray"){
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat edges;
            cv::Mat prewittX, prewittY;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2])/3;
                    imagecvin.at<float>(i, j) = var;
                }
            }
            cv::Mat kernelX = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
            cv::filter2D(imagecvin, prewittX, -1, kernelX);

            cv::Mat kernelY = (cv::Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
            cv::filter2D(imagecvin, prewittY, -1, kernelY);

            cv::magnitude(prewittX, prewittY, imagecvout);

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
        if(mode=="roberts_threshold"){
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat edges;
            cv::Mat prewittX, prewittY;
            cv::Mat magnitude;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2])/3;
                    imagecvin.at<float>(i, j) = var;
                }
            }
            cv::Mat kernelX = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
            cv::filter2D(imagecvin, prewittX, -1, kernelX);

            cv::Mat kernelY = (cv::Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
            cv::filter2D(imagecvin, prewittY, -1, kernelY);

            cv::magnitude(prewittX, prewittY, magnitude);

            cv::threshold(magnitude, imagecvout, threshold, maxThreshold, cv::THRESH_BINARY);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
        if(mode=="canny_gray"){
            cv::Mat imagecvin(h, w, CV_8U);
            cv::Mat imagecvout(h, w, CV_8U);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    imagecvin.at<uchar>(i, j) = int(rgb[0] * 255);
                }
            }
            // 自适应阈值化
            cv::Mat thresholdImage;
            int maxValue = 255;  // 最大像素值
            int blockSize = 3;  // 邻域块大小
            double C = 2;  // 常数项
            cv::adaptiveThreshold(imagecvin, thresholdImage, maxValue, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);

            // 执行 Canny 边缘检测
            int apertureSize = 3;  // 孔径大小，默认为 3
            bool L2gradient = false;  // 使用 L1 范数计算梯度幅值
            cv::Canny(thresholdImage, imagecvout, 0, 0, apertureSize, L2gradient);

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<uchar>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
        if(mode=="canny_threshold"){
            cv::Mat imagecvin(h, w, CV_8U);
            cv::Mat imagecvout(h, w, CV_8U);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    imagecvin.at<uchar>(i, j) = int(rgb[0] * 255);
                }
            }
            cv::Canny(imagecvin,imagecvout,threshold, maxThreshold);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<uchar>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
    }
};
ZENDEFNODE(EdgeDetect, {
    {
        {"image"},
        {"enum zeno_gray zeno_threshold sobel_gray sobel_threshold roberts_gray roberts_threshold prewitt_gray prewitt_threshold canny_gray canny_threshold", "mode", "sobel_gray"},
        {"float", "threshold", "50"},
        {"float", "maxThreshold", "9999"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
});

struct EdgeDetect_DIY: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        auto ktop = get_input2<vec3f>("kerneltop");
        auto kmid = get_input2<vec3f>("kernelmid");
        auto kbot = get_input2<vec3f>("kernelbot");
        int threshold = get_input2<int>("threshold");
        int maxThreshold = get_input2<int>("maxThreshold");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        if(mode=="diy_gray"){
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat edges;
            cv::Mat prewittX, prewittY;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2])/3;
                    imagecvin.at<float>(i, j) = var;
                }
            }
            cv::Mat kernelX = (cv::Mat_<float>(3, 3) << ktop[0], kmid[0], kbot[0],
                                                        ktop[1], kmid[1], kbot[1],
                                                        ktop[2], kmid[2], kbot[2]);
            cv::filter2D(imagecvin, prewittX, -1, kernelX);

            cv::Mat kernelY = (cv::Mat_<float>(3, 3) <<  ktop[0], kmid[1], kbot[2],
                                                         kmid[0], kmid[1], kbot[2],
                                                         kbot[0], kmid[1], kbot[2]);
            cv::filter2D(imagecvin, prewittY, -1, kernelY);
            cv::magnitude(prewittX, prewittY, imagecvout);

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }

        if(mode=="diy_threshold"){
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat edges;
            cv::Mat prewittX, prewittY;
            cv::Mat magnitude;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2])/3;
                    imagecvin.at<float>(i, j) = var;
                }
            }
            cv::Mat kernelX = (cv::Mat_<float>(3, 3) << ktop[0], kmid[0], kbot[0],
                                                        ktop[1], kmid[1], kbot[1],
                                                        ktop[2], kmid[2], kbot[2]);
            cv::filter2D(imagecvin, prewittX, -1, kernelX);

            cv::Mat kernelY = (cv::Mat_<float>(3, 3) <<  ktop[0], kmid[1], kbot[2],
                                                         kmid[0], kmid[1], kbot[2],
                                                         kbot[0], kmid[1], kbot[2]);
            cv::filter2D(imagecvin, prewittY, -1, kernelY);

            cv::magnitude(prewittX, prewittY, magnitude);

            cv::threshold(magnitude, imagecvout, threshold, maxThreshold, cv::THRESH_BINARY);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
    }
};
ZENDEFNODE(EdgeDetect_DIY, {
    {
        {"image"},
        {"enum diy_gray diy_threshold", "mode", "diy_gray"},
        {"vec3f", "kerneltop", "-1,-2,-1"},
        {"vec3f", "kernelmid", "0,0,0"},
        {"vec3f", "kernelbot", "1,2,1"},
        {"float", "threshold", "50"},
        {"float", "maxThreshold", "9999"},
    },
    {
        {"image"}
    },
    {},
    { "deprecated" },
});

struct EdgeDetect_Sobel : INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        int threshold = get_input2<int>("threshold");
        int maxThreshold = get_input2<int>("maxThreshold");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        if(mode=="sobel_gray"){
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
            cv::Mat gradX, gradY;
            cv::Sobel(imagecvin, gradX, CV_32F, 1, 0);
            cv::Sobel(imagecvin, gradY, CV_32F, 0, 1);
            cv::Mat gradientMagnitude, gradientDirection;
            cv::cartToPolar(gradX, gradY, imagecvout, gradientDirection, true);

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }

        if(mode=="sobel_threshold"){
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
            cv::Mat gradX, gradY;
            cv::Sobel(imagecvin, gradX, CV_32F, 1, 0);
            cv::Sobel(imagecvin, gradY, CV_32F, 0, 1);
            cv::Mat gradientMagnitude, gradientDirection;
            cv::cartToPolar(gradX, gradY, gradientMagnitude, gradientDirection, true);
            cv::threshold(gradientMagnitude, imagecvout, threshold, maxThreshold, cv::THRESH_BINARY);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
    }
};
ZENDEFNODE(EdgeDetect_Sobel, {
    {
        {"image"},
        {"enum sobel_gray sobel_threshold", "mode", "sobel_gray"},
        {"float", "threshold", "50"},
        {"float", "maxThreshold", "9999"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
});

struct EdgeDetect_Roberts: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        int threshold = get_input2<int>("threshold");
        int maxThreshold = get_input2<int>("maxThreshold");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        if(mode=="roberts_gray"){
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat robertsX, robertsY;
            cv::Mat magnitude;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2])/3;
                    imagecvin.at<float>(i, j) = var;
                }
            }
            cv::Mat kernelX = (cv::Mat_<float>(2, 2) << 1, 0, 0, -1);
            cv::filter2D(imagecvin, robertsX, -1, kernelX);
            cv::Mat kernelY = (cv::Mat_<float>(2, 2) << 0, 1, -1, 0);
            cv::filter2D(imagecvin, robertsY, -1, kernelY);
            cv::magnitude(robertsX, robertsY, imagecvout);

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }

        if(mode=="roberts_threshold"){
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat robertsX, robertsY;
            cv::Mat magnitude;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2])/3;
                    imagecvin.at<float>(i, j) = var;
                }
            }
            cv::Mat kernelX = (cv::Mat_<float>(2, 2) << 1, 0, 0, -1);
            cv::filter2D(imagecvin, robertsX, -1, kernelX);
            cv::Mat kernelY = (cv::Mat_<float>(2, 2) << 0, 1, -1, 0);
            cv::filter2D(imagecvin, robertsY, -1, kernelY);
            cv::magnitude(robertsX, robertsY, magnitude);
            cv::threshold(magnitude, imagecvout, threshold, maxThreshold, cv::THRESH_BINARY);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
    }
};
ZENDEFNODE(EdgeDetect_Roberts, {
    {
        {"image"},
        {"enum roberts_gray roberts_threshold", "mode", "roberts_gray"},
        {"float", "threshold", "50"},
        {"float", "maxThreshold", "255"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
});

struct EdgeDetect_Prewitt: INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        int threshold = get_input2<int>("threshold");
        int maxThreshold = get_input2<int>("maxThreshold");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        if(mode=="roberts_gray"){
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat edges;
            cv::Mat prewittX, prewittY;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2])/3;
                    imagecvin.at<float>(i, j) = var;
                }
            }
            cv::Mat kernelX = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
            cv::filter2D(imagecvin, prewittX, -1, kernelX);

            cv::Mat kernelY = (cv::Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
            cv::filter2D(imagecvin, prewittY, -1, kernelY);

            cv::magnitude(prewittX, prewittY, imagecvout);

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }

        if(mode=="roberts_threshold"){
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat edges;
            cv::Mat prewittX, prewittY;
            cv::Mat magnitude;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2])/3;
                    imagecvin.at<float>(i, j) = var;
                }
            }
            cv::Mat kernelX = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
            cv::filter2D(imagecvin, prewittX, -1, kernelX);

            cv::Mat kernelY = (cv::Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
            cv::filter2D(imagecvin, prewittY, -1, kernelY);

            cv::magnitude(prewittX, prewittY, magnitude);

            cv::threshold(magnitude, imagecvout, threshold, maxThreshold, cv::THRESH_BINARY);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<float>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
    }
};
ZENDEFNODE(EdgeDetect_Prewitt, {
    {
        {"image"},
        {"enum prewitt_gray prewitt_threshold", "mode", "prewitt_gray"},
        {"float", "threshold", "50"},
        {"float", "maxThreshold", "255"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
});

//边缘检测
struct EdgeDetect_Canny : INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        int threshold1 = get_input2<int>("threshold1");
        int threshold2 = get_input2<int>("threshold2");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        if(mode=="canny_gray"){
            cv::Mat imagecvin(h, w, CV_8U);
            cv::Mat imagecvout(h, w, CV_8U);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    imagecvin.at<uchar>(i, j) = int(rgb[0] * 255);
                }
            }
            // 自适应阈值化
            cv::Mat thresholdImage;
            int maxValue = 255;  // 最大像素值
            int blockSize = 3;  // 邻域块大小
            double C = 2;  // 常数项
            cv::adaptiveThreshold(imagecvin, thresholdImage, maxValue, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);

            // 执行 Canny 边缘检测
            int apertureSize = 3;  // 孔径大小，默认为 3
            bool L2gradient = false;  // 使用 L1 范数计算梯度幅值
            cv::Canny(thresholdImage, imagecvout, 0, 0, apertureSize, L2gradient);

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<uchar>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
        if(mode=="canny_threshold"){
            cv::Mat imagecvin(h, w, CV_8U);
            cv::Mat imagecvout(h, w, CV_8U);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    imagecvin.at<uchar>(i, j) = int(rgb[0] * 255);
                }
            }
            cv::Canny(imagecvin, imagecvout, threshold1, threshold2, 3, false);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float r = float(imagecvout.at<uchar>(i, j)) / 255.f;
                    image->verts[i * w + j] = {r, r, r};
                }
            }
            set_output("image", image);
        }
    }
};
ZENDEFNODE(EdgeDetect_Canny, {
    {
        {"image"},
        {"enum canny_gray canny_threshold", "mode", "canny_gray"},
        {"float", "threshold1", "100"},
        {"float", "threshold2", "200"},
    },
    {
        {"image"}
    },
    {},
    { "image" },
});

struct ImageExtractFeature_ORB : INode {
    void apply() override {
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
                imagecvin.at<cv::Vec3b>(i, j) = (255 * rgb[0], 255 * rgb[1], 255 * rgb[2]);
            }
        }
        cv::cvtColor(imagecvin, imagecvgray, cv::COLOR_BGR2GRAY);
//        imagecvin.convertTo(imagecvgray, CV_8U);
        orb->detect(imagecvgray, keypoints);
        zeno::log_info("orb->detect (imagecvin keypoints:{})", keypoints.size());
        orb->compute(imagecvgray, keypoints, imagecvdetect);
        zeno::log_info("orb->compute(imagecvin, keypoints, imagecvout)");
//        cv::drawKeypoints(imagecvin, keypoints, imagecvout, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

        cv::drawKeypoints(imagecvin, keypoints, imagecvout, cv::Scalar(0, 0, 255),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
        zeno::log_info(
                "cv::drawKeypoints(imagecvin, keypoints, imagecvout, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);");
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
//                cv::Vec3f rgb = imagecvout.at<cv::Vec3f>(i, j);
//                image->verts[i * w + j] = {rgb/255, rgb/255, rgb/255};
//                image->verts[i * w + j] = {rgb[0], rgb[1], rgb[2]};
                image->verts[i * w + j][0] = imagecvin.at<cv::Vec3b>(i, j)[0];
                image->verts[i * w + j][1] = imagecvin.at<cv::Vec3b>(i, j)[1];
                image->verts[i * w + j][2] = imagecvin.at<cv::Vec3b>(i, j)[2];

            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageExtractFeature_ORB, {
    {
        { "image" },
    },
    {
        { "image" },
    },
    {},
    { "image" },
});

struct ImageExtractFeature_SIFT : INode {
    void apply() override {
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
        { "image" },
    },
    {
        { "PrimitiveObject", "image" },
    },
    {},
    { "image" },
});
}
}