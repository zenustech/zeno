#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/scope_exit.h>
#include <cmath>
#include <zeno/utils/log.h>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/stitching.hpp>
#include <zeno/types/ListObject.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <set>
#include <algorithm>
#include <vector>
#include <zeno/types/MatrixObject.h>
#include "imgcv.h"
#include <variant>

using namespace cv;

namespace zeno {

namespace {

static void zenoedge(std::shared_ptr<PrimitiveObject> &grayImage, int width, int height, std::vector<float> &dx,
                     std::vector<float> &dy) {
    dx.resize(width * height);
    dy.resize(width * height);
#pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float gx =
                    -grayImage->verts[(y - 1) * width + x - 1][0] + grayImage->verts[(y - 1) * width + x + 1][0]
                    - 2.0f * grayImage->verts[y * width + x - 1][0] +
                    2.0f * grayImage->verts[y * width + x + 1][0]
                    - grayImage->verts[(y + 1) * width + x - 1][0] +
                    grayImage->verts[(y + 1) * width + x + 1][0];
            float gy = grayImage->verts[(y - 1) * width + x - 1][0] +
                       2.0f * grayImage->verts[(y - 1) * width + x][0] +
                       grayImage->verts[(y - 1) * width + x + 1][0]
                       - grayImage->verts[(y + 1) * width + x - 1][0] -
                       2.0f * grayImage->verts[(y + 1) * width + x][0] -
                       grayImage->verts[(y + 1) * width + x + 1][0];

            dx[y * width + x] = gx;
            dy[y * width + x] = gy;
        }
    }
}

// 计算法向量
static void normalMap(std::shared_ptr<PrimitiveObject> &grayImage, int width, int height, std::vector<float> &normal) {
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

static void scharr2(std::shared_ptr<PrimitiveObject> &src, std::shared_ptr<PrimitiveObject> &dst, int width, int height,
        int threshold) {
    std::vector<int> gx(width * height);
    std::vector<int> gy(width * height);
    dst->verts.resize(width * height);

    // Calculate gradients
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            gx[idx] = (-3 * src->verts[(y - 1) * width + x - 1][0] - 10 * src->verts[y * width + x - 1][0] -
                       3 * src->verts[(y + 1) * width + x - 1][0] +
                       3 * src->verts[(y - 1) * width + x + 1][0] + 10 * src->verts[y * width + x + 1][0] +
                       3 * src->verts[(y + 1) * width + x + 1][0]);
            gy[idx] = (-3 * src->verts[(y - 1) * width + x - 1][0] - 10 * src->verts[(y - 1) * width + x][0] -
                       3 * src->verts[(y - 1) * width + x + 1][0] +
                       3 * src->verts[(y + 1) * width + x - 1][0] + 10 * src->verts[(y + 1) * width + x][0] +
                       3 * src->verts[(y + 1) * width + x + 1][0]);
        }
    }
    // Calculate gradient magnitude
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            int mag = std::sqrt(gx[idx] * gx[idx] + gy[idx] * gy[idx]);
            if (mag * 255 > threshold) {
                dst->verts[idx] = {1, 1, 1};
            } else {
                dst->verts[idx] = {0, 0, 0};
            }
            float g = std::min(1, std::max(0, mag));
            dst->verts[idx] = {g, g, g};
        }
    }
}

struct ImageEdgeDetect : INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        int threshold = get_input2<int>("threshold");
        int maxThreshold = get_input2<int>("maxThreshold");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        if (mode == "zeno_gray") {
            std::vector<float> dx, dy;
            zenoedge(image, w, h, dx, dy);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {

                    float gradient = std::sqrt(pow(dx[i * w + j], 2) + pow(dy[i * w + j], 2));
                    image->verts[i * w + j] = {gradient, gradient, gradient};
                }
            }
            set_output("image", image);
        }

        if (mode == "zeno_threshold") {
            std::vector<float> dx, dy;
            zenoedge(image, w, h, dx, dy);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {

                    float gradient = std::sqrt(pow(dx[i * w + j], 2) + pow(dy[i * w + j], 2));

                    if (gradient * 255 > threshold) {
                        image->verts[i * w + j] = {1, 1, 1};
                    } else {
                        image->verts[i * w + j] = {0, 0, 0};
                    }
                }
            }
            set_output("image", image);
        }
        if (mode == "sobel_gray") {
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2]) / 3;
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
        if (mode == "sobel_threshold") {
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2]) / 3;
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
        if (mode == "roberts_gray") {
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat robertsX, robertsY;
            cv::Mat magnitude;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2]) / 3;
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

        if (mode == "roberts_threshold") {
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat robertsX, robertsY;
            cv::Mat magnitude;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2]) / 3;
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
        if (mode == "roberts_gray") {
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat edges;
            cv::Mat prewittX, prewittY;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2]) / 3;
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
        if (mode == "roberts_threshold") {
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat edges;
            cv::Mat prewittX, prewittY;
            cv::Mat magnitude;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2]) / 3;
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
        if (mode == "canny_gray") {
            cv::Mat imagecvin(h, w, CV_8U);
            cv::Mat imagecvout(h, w, CV_8U);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    imagecvin.at<uchar>(i, j) = int(rgb[0] * 255);
                }
            }
            cv::Mat thresholdImage;
            int maxValue = 255;  // 最大像素值
            int blockSize = 3;  // 邻域块大小
            double C = 2;
            cv::adaptiveThreshold(imagecvin, thresholdImage, maxValue, cv::ADAPTIVE_THRESH_MEAN_C,
                                  cv::THRESH_BINARY, blockSize, C);

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
        if (mode == "canny_threshold") {
            cv::Mat imagecvin(h, w, CV_8U);
            cv::Mat imagecvout(h, w, CV_8U);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    imagecvin.at<uchar>(i, j) = int(rgb[0] * 255);
                }
            }
            cv::Canny(imagecvin, imagecvout, threshold, maxThreshold);
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

ZENDEFNODE(ImageEdgeDetect, {
    {
        { "image" },
        { "enum zeno_gray zeno_threshold sobel_gray sobel_threshold roberts_gray roberts_threshold prewitt_gray prewitt_threshold canny_gray canny_threshold", "mode", "sobel_gray" },
        { "float", "threshold", "50" },
        { "float", "maxThreshold", "9999" },
    },
    {
        { "image" }
    },
    {},
    { "image" },
});

struct ImageEdgeDetectDIY : INode {
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

        if (mode == "diy_gray") {
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat edges;
            cv::Mat prewittX, prewittY;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2]) / 3;
                    imagecvin.at<float>(i, j) = var;
                }
            }
            cv::Mat kernelX = (cv::Mat_<float>(3, 3) << ktop[0], kmid[0], kbot[0],
                    ktop[1], kmid[1], kbot[1],
                    ktop[2], kmid[2], kbot[2]);
            cv::filter2D(imagecvin, prewittX, -1, kernelX);

            cv::Mat kernelY = (cv::Mat_<float>(3, 3) << ktop[0], kmid[1], kbot[2],
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

        if (mode == "diy_threshold") {
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat edges;
            cv::Mat prewittX, prewittY;
            cv::Mat magnitude;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2]) / 3;
                    imagecvin.at<float>(i, j) = var;
                }
            }
            cv::Mat kernelX = (cv::Mat_<float>(3, 3) << ktop[0], kmid[0], kbot[0],
                    ktop[1], kmid[1], kbot[1],
                    ktop[2], kmid[2], kbot[2]);
            cv::filter2D(imagecvin, prewittX, -1, kernelX);

            cv::Mat kernelY = (cv::Mat_<float>(3, 3) << ktop[0], kmid[1], kbot[2],
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

ZENDEFNODE(ImageEdgeDetectDIY, {
    {
        { "image" },
        { "enum diy_gray diy_threshold", "mode", "diy_gray" },
        { "vec3f", "kerneltop", "-1,-2,-1" },
        { "vec3f", "kernelmid", "0,0,0" },
        { "vec3f", "kernelbot", "1,2,1" },
        { "float", "threshold", "50" },
        { "float", "maxThreshold", "9999" },
    },
    {
        { "image" }
    },
    {},
    { "deprecated" },
});

struct ImageEdgeDetectSobel : INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        int threshold = get_input2<int>("threshold");
        int maxThreshold = get_input2<int>("maxThreshold");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        if (mode == "sobel_gray") {
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2]) / 3;
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

        if (mode == "sobel_threshold") {
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2]) / 3;
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

ZENDEFNODE(ImageEdgeDetectSobel, {
    {
        { "image" },
        { "enum sobel_gray sobel_threshold", "mode", "sobel_gray" },
        { "float", "threshold", "50" },
        { "float", "maxThreshold", "9999" },
    },
    {
        { "image" }
    },
    {},
    { "image" },
});

struct ImageEdgeDetectRoberts : INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        int threshold = get_input2<int>("threshold");
        int maxThreshold = get_input2<int>("maxThreshold");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        if (mode == "roberts_gray") {
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat robertsX, robertsY;
            cv::Mat magnitude;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2]) / 3;
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

        if (mode == "roberts_threshold") {
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat robertsX, robertsY;
            cv::Mat magnitude;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2]) / 3;
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

ZENDEFNODE(ImageEdgeDetectRoberts, {
    {
        { "image" },
        { "enum roberts_gray roberts_threshold", "mode", "roberts_gray" },
        { "float", "threshold", "50" },
        { "float", "maxThreshold", "255" },
    },
    {
        { "image" }
    },
    {},
    { "image" },
});

struct ImageEdgeDetectPrewitt : INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        int threshold = get_input2<int>("threshold");
        int maxThreshold = get_input2<int>("maxThreshold");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        if (mode == "roberts_gray") {
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat edges;
            cv::Mat prewittX, prewittY;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2]) / 3;
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

        if (mode == "roberts_threshold") {
            cv::Mat imagecvin(h, w, CV_32F);
            cv::Mat imagecvout(h, w, CV_32F);
            cv::Mat edges;
            cv::Mat prewittX, prewittY;
            cv::Mat magnitude;
            int var = 1;
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    var = 255 * (rgb[0] + rgb[1] + rgb[2]) / 3;
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

ZENDEFNODE(ImageEdgeDetectPrewitt, {
    {
        { "image" },
        { "enum prewitt_gray prewitt_threshold", "mode", "prewitt_gray" },
        { "float", "threshold", "50" },
        { "float", "maxThreshold", "255" },
    },
    {
        { "image" }
    },
    {},
    { "image" },
});

struct ImageEdgeDetectCanny : INode {
    void apply() override {
        std::shared_ptr<PrimitiveObject> image = get_input2<PrimitiveObject>("image");
        auto mode = get_input2<std::string>("mode");
        int threshold1 = get_input2<int>("threshold1");
        int threshold2 = get_input2<int>("threshold2");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");

        if (mode == "canny_gray") {
            cv::Mat imagecvin(h, w, CV_8U);
            cv::Mat imagecvout(h, w, CV_8U);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    vec3f rgb = image->verts[i * w + j];
                    imagecvin.at<uchar>(i, j) = int(rgb[0] * 255);
                }
            }
            cv::Mat thresholdImage;
            int maxValue = 255;  // 最大像素值
            int blockSize = 3;  // 邻域块大小
            double C = 2;
            cv::adaptiveThreshold(imagecvin, thresholdImage, maxValue, cv::ADAPTIVE_THRESH_MEAN_C,
                                  cv::THRESH_BINARY, blockSize, C);

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
        if (mode == "canny_threshold") {
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

ZENDEFNODE(ImageEdgeDetectCanny, {
    {
        { "image" },
        { "enum canny_gray canny_threshold", "mode", "canny_gray" },
        { "float", "threshold1", "100" },
        { "float", "threshold2", "200" },
    },
    {
        { "image" }
    },
    {},
    { "image" },
});

struct ImageStitching : INode {
    void apply() override {
        auto image1 = get_input<PrimitiveObject>("image1");
        auto image2 = get_input<PrimitiveObject>("image2");
        auto &ud1 = image1->userData();
        int w1 = ud1.get2<int>("w");
        int h1 = ud1.get2<int>("h");
        auto &ud2 = image2->userData();
        int w2 = ud2.get2<int>("w");
        int h2 = ud2.get2<int>("h");
        cv::Mat imagecvin1(h1, w1, CV_8UC3);
        cv::Mat imagecvin2(h2, w2, CV_8UC3);
        cv::Mat descriptors1, descriptors2;
        std::vector<cv::KeyPoint> ikeypoints;
        for (int i = 0; i < h1; i++) {
            for (int j = 0; j < w1; j++) {
                vec3f rgb1 = image1->verts[i * w1 + j];
                cv::Vec3b& pixel = imagecvin1.at<cv::Vec3b>(i, j);
                pixel[0] = rgb1[0] * 255;
                pixel[1] = rgb1[1] * 255;
                pixel[2] = rgb1[2] * 255;
            }
        }
        for (int i = 0; i < h2; i++) {
            for (int j = 0; j < w2; j++) {
                vec3f rgb2 = image2->verts[i * w2 + j];
                cv::Vec3b& pixel = imagecvin2.at<cv::Vec3b>(i, j);
                pixel[0] = rgb2[0] * 255;
                pixel[1] = rgb2[1] * 255;
                pixel[2] = rgb2[2] * 255;
            }
        }
        std::vector<cv::Mat> images;
        images.push_back(imagecvin1);
        images.push_back(imagecvin2);
        cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create();
        cv::Mat result;
        cv::Stitcher::Status status = stitcher->stitch(images, result);
        if (status == cv::Stitcher::OK) {
            int w = result.cols;
            int h = result.rows;
            image1->verts.resize(w*h);
            ud1.set2("w",w);
            ud1.set2("h",h);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                    image1->verts[i * w + j][0] = static_cast<float>(pixel[0])/255;
                    image1->verts[i * w + j][1] = static_cast<float>(pixel[1])/255;
                    image1->verts[i * w + j][2] = static_cast<float>(pixel[2])/255;
                }
            }
        } else {
            zeno::log_info("stitching failed");
        }
        set_output("image", image1);
    }
};

ZENDEFNODE(ImageStitching, {
    {
        {"image1" },
        {"image2" },
    },
    {
        { "image" },
    },
    {},
    { "image" },
});

//struct ImageStitchingList: INode {
//    void apply() override {
//        auto primList = get_input<ListObject>("listPrim")->getRaw<PrimitiveObject>();
//        auto mode = get_input2<std::string>("mode");
//        auto &ud1 = image1->userData();
//        int w1 = ud1.get2<int>("w");
//        int h1 = ud1.get2<int>("h");
////        auto &ud2 = image2->userData();
////        int w2 = ud2.get2<int>("w");
////        int h2 = ud2.get2<int>("h");
//        auto nFeatures = get_input2<float>("nFeatures");
//        auto scaleFactor = get_input2<float>("scaleFactor");
//        auto edgeThreshold = get_input2<float>("edgeThreshold");
//        auto patchSize = get_input2<float>("patchSize");
//        cv::Mat imagecvin1(h1, w1, CV_8UC3);
////        cv::Mat imagecvin2(h2, w2, CV_8UC3);
//        cv::Mat descriptors1, descriptors2;
//        std::vector<cv::KeyPoint> ikeypoints;
//        for (int i = 0; i < h1; i++) {
//            for (int j = 0; j < w1; j++) {
//                vec3f rgb1 = image1->verts[i * w1 + j];
//                cv::Vec3b& pixel = imagecvin1.at<cv::Vec3b>(i, j);
//                pixel[0] = rgb1[0] * 255;
//                pixel[1] = rgb1[1] * 255;
//                pixel[2] = rgb1[2] * 255;
//            }
//        }
//        for (int i = 0; i < h2; i++) {
//            for (int j = 0; j < w2; j++) {
//                vec3f rgb2 = image2->verts[i * w2 + j];
//                cv::Vec3b& pixel = imagecvin2.at<cv::Vec3b>(i, j);
//                pixel[0] = rgb2[0] * 255;
//                pixel[1] = rgb2[1] * 255;
//                pixel[2] = rgb2[2] * 255;
//            }
//        }
//        std::vector<cv::Mat> images;
//        images.push_back(imagecvin1);
//        images.push_back(imagecvin2);
//        cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create();
//        cv::Mat result;
//        cv::Stitcher::Status status = stitcher->stitch(images, result);
//        if (status == cv::Stitcher::OK) {
//            int w = result.cols;
//            int h = result.rows;
//            image1->verts.resize(w*h);
//            ud1.set2("w",w);
//            ud1.set2("h",h);
//            for (int i = 0; i < h; i++) {
//                for (int j = 0; j < w; j++) {
//                    cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
//                    image1->verts[i * w + j][0] = static_cast<float>(pixel[0])/255;
//                    image1->verts[i * w + j][1] = static_cast<float>(pixel[1])/255;
//                    image1->verts[i * w + j][2] = static_cast<float>(pixel[2])/255;
//                }
//            }
//        } else {
//            zeno::log_info("stitching failed");
//        }
//        set_output("image", image1);
//    }
//};
//
//ZENDEFNODE(ImageStitchingList, {
//    {
//        {"list", "listPrim"},
//    },
//    {
//        { "image" },
//    },
//    {},
//    { "image" },
//});
struct ImageFeatureDetectORB : INode {
    void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto visualize = get_input2<bool>("visualize");
        auto nFeatures = get_input2<float>("nFeatures");
        auto scaleFactor = get_input2<float>("scaleFactor");
        auto edgeThreshold = get_input2<float>("edgeThreshold");
        auto patchSize = get_input2<float>("patchSize");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        auto &dp = image->uvs.add_attr<float>("descriptors"); //descriptors.size() = 32 * nFeatures
        auto &kp = image->uvs.add_attr<vec2f>("keypoints");
        auto &pos = image->verts.attr<vec3f>("pos");
        cv::Ptr<cv::ORB> orb = cv::ORB::create(nFeatures, scaleFactor, 8, edgeThreshold, 0, 2,
                                               cv::ORB::HARRIS_SCORE, 31, patchSize);
        cv::Mat imagecvin(h, w, CV_8UC3);
        cv::Mat imagecvgray(h, w, CV_8U);
        cv::Mat imagecvout(h, w, CV_8UC3);
        std::vector<cv::KeyPoint> ikeypoints;

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                vec3f rgb = image->verts[i * w + j];
                cv::Vec3b& pixel = imagecvin.at<cv::Vec3b>(i, j);
                pixel[0] = rgb[0] * 255;
                pixel[1] = rgb[1] * 255;
                pixel[2] = rgb[2] * 255;
            }
        }
        cv::cvtColor(imagecvin, imagecvgray, cv::COLOR_RGB2GRAY);
        cv::Mat idescriptor;
        orb->detectAndCompute(imagecvgray, cv::noArray(), ikeypoints, idescriptor);
        if (ikeypoints.size() == 0) {
            throw zeno::Exception("Did not find any features");
        }
        cv::Mat imageFloat;
        idescriptor.convertTo(imageFloat, CV_32F, 1.0 / 255.0);
        cv::Size ds = imageFloat.size();
        int dss = ds.width * ds.height;
        image->uvs.resize(dss);
        image->userData().set2("dw",ds.width);
        zeno::log_info("orbDescriptor.width:{}, orbDescriptor.height:{}",ds.width, ds.height);
        for (int i = 0; i < dss; i++) {
            dp[i] = imageFloat.at<float>(i);
        }
        for(size_t i = 0;i < ikeypoints.size();i++){
            cv::KeyPoint keypoint = ikeypoints[i];
            float x = static_cast<float>(keypoint.pt.x);
            float y = static_cast<float>(keypoint.pt.y);
            kp[i] = {x, y};
        }

        if (visualize) {
            cv::drawKeypoints(imagecvin, ikeypoints, imagecvout, cv::Scalar(255, 0, 0),
                              cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            // | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    cv::Vec3b pixel = imagecvout.at<cv::Vec3b>(i, j);
                    image->verts[i * w + j][0] = zeno::min(static_cast<float>(pixel[0])/255,1.0f);
                    image->verts[i * w + j][1] = zeno::min(static_cast<float>(pixel[1])/255,1.0f);
                    image->verts[i * w + j][2] = zeno::min(static_cast<float>(pixel[2])/255,1.0f);
                }
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageFeatureDetectORB, {
    {
        { "image" },
        { "bool", "visualize", "1" },
        { "float", "nFeatures", "100" },
        { "float", "scaleFactor", "1.2" },
        { "float", "edgeThreshold", "20" },
        { "float", "patchSize", "10" },
    },
    {
        { "image" },
    },
    {},
    { "image" },
});

struct ImageFeatureDetectSIFT : INode {
    void apply() override {
        auto image = get_input<PrimitiveObject>("image");
        auto visualize = get_input2<bool>("visualize");
        auto nFeatures = get_input2<float>("nFeatures");
        auto nOctaveLayers = get_input2<float>("nOctaveLayers");
        auto contrastThreshold = get_input2<float>("contrastThreshold");
        auto edgeThreshold = get_input2<float>("edgeThreshold");
        auto sigma = get_input2<float>("sigma");
        auto &ud = image->userData();
        int w = ud.get2<int>("w");
        int h = ud.get2<int>("h");
        auto &dp = image->uvs.add_attr<float>("descriptors");
        auto &kp = image->uvs.add_attr<vec2f>("keypoints");
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create(nFeatures, nOctaveLayers,
                                                  contrastThreshold, edgeThreshold,sigma);

        cv::Mat imagecvin(h, w, CV_8UC3);
        cv::Mat imagecvgray(h, w, CV_8U);
        cv::Mat idescriptor;
        cv::Mat imagecvout(h, w, CV_8UC3);
        std::vector<cv::KeyPoint> ikeypoints;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                vec3f rgb = image->verts[i * w + j];
                cv::Vec3b& pixel = imagecvin.at<cv::Vec3b>(i, j);
                pixel[0] = rgb[0] * 255;
                pixel[1] = rgb[1] * 255;
                pixel[2] = rgb[2] * 255;
            }
        }
        cv::cvtColor(imagecvin, imagecvgray, cv::COLOR_RGB2GRAY);
        sift->detectAndCompute(imagecvgray, cv::noArray(), ikeypoints, idescriptor);
        if (ikeypoints.size() == 0) {
            throw zeno::Exception("Did not find any features");
        }
        cv::drawKeypoints(imagecvin, ikeypoints, imagecvout, cv::Scalar(255, 0, 0),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//                          | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG

        cv::Mat imageFloat;
        idescriptor.convertTo(imageFloat, CV_32F, 1.0 / 255.0);
        cv::Size ds = imageFloat.size();
        int dss = ds.width * ds.height;
        image->userData().set2("dw",ds.width);
        image->uvs.resize(dss);
        zeno::log_info("siftDescriptor.width:{}, siftDescriptor.height:{}",ds.width, ds.height);
        for (int i = 0; i < dss; i++) {
            dp[i] = imageFloat.at<float>(i);
        }
        for(size_t i = 0;i < ikeypoints.size();i++){
            cv::KeyPoint keypoint = ikeypoints[i];
            float x = static_cast<float>(keypoint.pt.x);
            float y = static_cast<float>(keypoint.pt.y);
            kp[i] = {x, y};
        }
        if (visualize) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    cv::Vec3b pixel = imagecvout.at<cv::Vec3b>(i, j);
                    image->verts[i * w + j][0] = zeno::min(static_cast<float>(pixel[0])/255,1.0f);
                    image->verts[i * w + j][1] = zeno::min(static_cast<float>(pixel[1])/255,1.0f);
                    image->verts[i * w + j][2] = zeno::min(static_cast<float>(pixel[2])/255,1.0f);
                }
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ImageFeatureDetectSIFT, {
    {
        { "image" },
        { "bool", "visualize", "1" },
        { "float", "nFeatures", "0" }, //所需的关键点数量
        { "float", "nOctaveLayers", "3" }, //每组尺度层级的层数
        { "float", "contrastThreshold", "0.04" }, //关键点提取的对比度阈值
        { "float", "edgeThreshold", "10" }, //关键点提取的边缘阈值
        { "float", "sigma", "1.0" }, //高斯滤波器的初始尺度
    },
    {
        { "PrimitiveObject", "image" },
    },
    {},
    { "image" },
});


struct ImageFeatureMatch : INode {
    void apply() override {
        auto image1 = get_input<PrimitiveObject>("image1");
        auto image2 = get_input<PrimitiveObject>("image2");
        auto matchD = get_input2<float>("maxMatchDistance");
        auto visualize = get_input2<bool>("visualize");
        auto stitch = get_input2<bool>("stitch");
        auto bpM = get_input2<bool>("perspectiveMatrix");
        auto bfM = get_input2<bool>("fundamentalMatrix");
        auto beM = get_input2<bool>("essentialMatrix");
        auto bhM = get_input2<bool>("homographyMatrix");
        cv::Mat OMatrix;

        auto &ud1 = image1->userData();
        int w1 = ud1.get2<int>("w");
        int h1 = ud1.get2<int>("h");
        auto &ud2 = image2->userData();
        int w2 = ud2.get2<int>("w");
        int h2 = ud2.get2<int>("h");

        auto image3 = std::make_shared<PrimitiveObject>();
        auto &ud3 = image3->userData();
        image3->verts.resize(w2 * h2);
        image3->uvs.resize(zeno::max(9,zeno::max(image1->uvs.size(),image2->uvs.size())));
        ud3.set2("h", h2);
        ud3.set2("w", w2);
        ud3.set2("isImage", 1);
        image3->verts = image2->verts;

        cv::Mat imagecvin1(h1, w1, CV_8UC3);
        cv::Mat imagecvin2(h2, w2, CV_8UC3);
        for (int i = 0; i < h1; i++) {
            for (int j = 0; j < w1; j++) {
                vec3f rgb = image1->verts[i * w1 + j];
                cv::Vec3b& pixel = imagecvin1.at<cv::Vec3b>(i, j);
                pixel[0] = rgb[0] * 255;
                pixel[1] = rgb[1] * 255;
                pixel[2] = rgb[2] * 255;
            }
        }
        for (int i = 0; i < h2; i++) {
            for (int j = 0; j < w2; j++) {
                vec3f rgb = image2->verts[i * w2 + j];
                cv::Vec3b& pixel = imagecvin2.at<cv::Vec3b>(i, j);
                pixel[0] = rgb[0] * 255;
                pixel[1] = rgb[1] * 255;
                pixel[2] = rgb[2] * 255;
            }
        }

        cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create();
        std::vector<cv::Mat> images;
        images.push_back(imagecvin1);
        images.push_back(imagecvin2);
        std::vector<cv::KeyPoint> keypoints;
        auto d1 = image1->uvs.add_attr<float>("descriptors");
        auto d2 = image2->uvs.add_attr<float>("descriptors");
        auto k1 = image1->uvs.add_attr<vec2f>("keypoints");
        auto k2 = image2->uvs.add_attr<vec2f>("keypoints");

        int dw1 = ud1.get2<int>("dw");
        int dw2 = ud2.get2<int>("dw");
        int ks1 = d1.size()/dw1;
        int ks2 = d2.size()/dw2;
        cv::Mat imagecvdescriptors1(ks1, dw1, CV_32F);
        cv::Mat imagecvdescriptors2(ks2, dw2, CV_32F);
        for (int i = 0; i < ks1; i++) {
            for (int j = 0; j < dw1; j++) {
                imagecvdescriptors1.at<float>(i, j) = d1[i * dw1 + j];
            }
        }
        for (int i = 0; i < ks2; i++) {
            for (int j = 0; j < dw2; j++) {
                imagecvdescriptors2.at<float>(i, j) = d2[i * dw2 + j];
            }
        }
        zeno::log_info("image1Keypoints.size:{},image2Keypoints.size:{},image1Descriptors.width:{},image2Descriptors.width:{}",ks1,ks2,dw1,dw2);

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
        std::vector<std::vector<cv::DMatch>> knnMatches;
        std::vector<cv::DMatch> filteredMatches;
        matcher->knnMatch(imagecvdescriptors1, imagecvdescriptors2, knnMatches, 2);
        filteredMatches.reserve(knnMatches.size());
//        auto &md = image3->uvs.add_attr<float>("matchDistance");
//        int mdi = 0;
        for (const auto& knnMatch : knnMatches) {
            if (knnMatch.size() < 2) {
                continue;
            }
            float distanceRatio = knnMatch[0].distance / knnMatch[1].distance;
            if (distanceRatio < matchD) {
                filteredMatches.push_back(knnMatch[0]);
//                md[mdi] = static_cast<float>(knnMatch[0].distance);
//                mdi++;
            }
        }
        zeno::log_info("knnMatches.size:{},filteredMatches.size：{}",knnMatches.size(),filteredMatches.size());
        std::vector<cv::Point2f> points1, points2;
        auto &m1 = image3->uvs.add_attr<vec3f>("image1MatchPoints");
        auto &m2 = image3->uvs.add_attr<vec3f>("image2MatchPoints");
        int m = 0;
        std::set<float> uniqueP1;
        for (const auto &match: filteredMatches) {
            if (uniqueP1.count(k1[match.queryIdx][0]) != 0) {
                continue;
            }
            uniqueP1.insert(k1[match.queryIdx][0]);
            cv::Point2f pt1(k1[match.queryIdx][0], k1[match.queryIdx][1]);
            cv::Point2f pt2(k2[match.trainIdx][0], k2[match.trainIdx][1]);
            points1.push_back(pt1);
            m1[m] = {static_cast<float>(match.queryIdx),k1[match.queryIdx][0],k1[match.queryIdx][1]};
            points2.push_back(pt2);
            m2[m] = {static_cast<float>(match.trainIdx),k2[match.trainIdx][0],k2[match.trainIdx][1]};
            m++;
        }
        image3->uvs.resize(m);

//cameraMatrix
        auto &cm = image3->uvs.add_attr<float>("cameraMatrix");
        cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << cm[0],0,cm[2],
                                                         0,cm[4],cm[5],
                                                         0, 0, 1);
        if(cm[0]==0){
            float fx = w1;   // image.width;
            float fy = h1;   // image.height;
            float cx = w1/2; // image.width / 2.0;
            float cy = h1/2; // image.height / 2.0;
            cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << fx,0,cx,
                    0,fy,cy,
                    0, 0, 1);

            cv::Size cs = cameraMatrix.size();
            int css = cs.width * cs.height;
            for (int i = 0; i < css; i++) {
                cm[i] = static_cast<float>(cameraMatrix.at<float>(i));
            }
        }

//perspectiveMatrix
        if(bpM){
            auto &pp1 = image3->uvs.add_attr<vec3f>("perspectiveImage1Points");
            auto &pp2 = image3->uvs.add_attr<vec3f>("perspectiveImage2Points");
            Point2f pPoints1[4] ,pPoints2[4];
            std::vector<vec3f> preTl;
            int pp = 0;
            std::set<float> uniquepP1;
            int filsize = filteredMatches.size();
            int foot = zeno::max(1,int(filsize/5));
            cv::Mat perspectiveMatrix;
            std::vector<cv::Mat> vmPerspectiveMatrix;
            std::vector<cv::Mat> svmPerspectiveMatrix;
            std::vector<double> vfPerspectiveTX;
            std::vector<double> vfPerspectiveTY;
            for(int group = 0;group < 7;group++){
                for (size_t i = 0; i < filsize; i+=foot) {
                    const cv::DMatch& match = filteredMatches[i];
                    float nkw1 = static_cast<float>(static_cast<int>(k1[match.queryIdx][0]));
                    if (uniquepP1.count(nkw1) != 0) {
                        continue;
                    }
                    if (pp==4) {
                        break;
                    }
                    uniquepP1.insert(nkw1);
                    float nkh1 = static_cast<float>(static_cast<int>(k1[match.queryIdx][1]));
                    float nkw2 = static_cast<float>(static_cast<int>(k2[match.trainIdx][0]));
                    float nkh2 = static_cast<float>(static_cast<int>(k2[match.trainIdx][1]));
                    pPoints1[pp].x = nkw1;
                    pPoints1[pp].y = nkh1;
                    pPoints2[pp].x = nkw2;
                    pPoints2[pp].y = nkh2;
                    pp1[pp] = {static_cast<float>(match.queryIdx),nkw1,nkh1};
                    pp2[pp] = {static_cast<float>(match.trainIdx),nkw2,nkh2};
                    pp++;
                }
                cv::Mat tempPerspectiveMatrix = cv::getPerspectiveTransform(pPoints1, pPoints2);
                vmPerspectiveMatrix.push_back(tempPerspectiveMatrix);
                vfPerspectiveTX.push_back(tempPerspectiveMatrix.at<double>(2));
            }
            std::sort(vfPerspectiveTX.begin(), vfPerspectiveTX.end());
            for (const cv::Mat& tperspectiveMatrix : vmPerspectiveMatrix) {
                if(tperspectiveMatrix.at<double>(2)==vfPerspectiveTX[2]||
                   tperspectiveMatrix.at<double>(2)==vfPerspectiveTX[3]||
                   tperspectiveMatrix.at<double>(2)==vfPerspectiveTX[4]){
                    svmPerspectiveMatrix.push_back(tperspectiveMatrix);
                }
            }
            for (const cv::Mat& tperspectiveMatrix : svmPerspectiveMatrix) {
                vfPerspectiveTY.push_back(tperspectiveMatrix.at<double>(5));
            }
            std::sort(vfPerspectiveTY.begin(), vfPerspectiveTY.end());
            for (const cv::Mat& tperspectiveMatrix : svmPerspectiveMatrix) {
                if(tperspectiveMatrix.at<double>(5)==vfPerspectiveTY[1]){
                    perspectiveMatrix = tperspectiveMatrix;
                }
            }

            auto &pm = image3->uvs.add_attr<float>("perspectiveMatrix");
            cv::Size ps = perspectiveMatrix.size();
            int pss = ps.width * ps.height;
            for (int i = 0; i < pss; i++) {
                pm[i] = static_cast<float>(perspectiveMatrix.at<double>(i));
            }
            auto &ptl = image3->uvs.add_attr<float>("perspectiveTranslation");
            auto &prt = image3->uvs.add_attr<float>("perspectiveRotation");

            ptl[0] = pm[2];
            ptl[1] = pm[5];
            ptl[2] = pm[8];

            prt[0] = pm[0];
            prt[1] = pm[1];
            prt[2] = pm[3];
            prt[3] = pm[4];
            OMatrix = perspectiveMatrix;
            zeno::log_info("perspectiveMatrix.width:{} perspectiveMatrix.height:{}",ps.width,ps.height);
        }

//fundamentalMatrix
        if(bfM){
            double fransacReprojThreshold = 3.0;
            double fconfidence = 0.99;
            int fmaxIters = 2000;
            cv::OutputArray fmask = cv::noArray();
            cv::Mat fundamentalMatrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC,
                                                               fransacReprojThreshold, fconfidence,fmaxIters,fmask);
            auto &fm = image3->uvs.add_attr<float>("fundamentalMatrix");
            cv::Size fs = fundamentalMatrix.size();
            int fss = fs.width * fs.height;
            for (int i = 0; i < fss; i++) {
                fm[i] = static_cast<float>(fundamentalMatrix.at<double>(i));
            }
            OMatrix = fundamentalMatrix;
            zeno::log_info("fundamentalMatrix.width:{} fundamentalMatrix.height:{}",fs.width,fs.height);
        }

//homographyMatrix
        if(bhM){
            cv::Mat homographyMatrix, hrotation, htranslation;
            int hmethod = cv::RANSAC;
            double hransacReprojThreshold = 3.0;
            const int hmaxIters = 2000;
            const double hconfidence = 0.995;
            homographyMatrix = cv::findHomography(points1,points2, hmethod,
                                                  hransacReprojThreshold,cv::noArray(),hmaxIters,hconfidence);
            auto &hm = image3->uvs.add_attr<float>("homographyMatrix");
            cv::Size hs = homographyMatrix.size();
            int hss = hs.width * hs.height;
            for (int i = 0; i < hss; i++) {
                if(-0.0001<homographyMatrix.at<float>(i)<0.0001){

                }
                if(1<homographyMatrix.at<float>(i)){

                }
                hm[i] = static_cast<float>(homographyMatrix.at<double>(i));
            }
            cv::recoverPose(homographyMatrix, points1,points2, cameraMatrix, hrotation, htranslation);
            auto &hrt = image3->uvs.add_attr<float>("homographyRotation");
            auto &htl = image3->uvs.add_attr<float>("homographyTranslation");
            cv::Size hts = htranslation.size();
            int htss = hts.width * hts.height;
            for (int i = 0; i < htss; i++) {
                htl[i] = static_cast<float>(htranslation.at<double>(i));
            }
            cv::Size hrs = hrotation.size();
            int hrss = hrs.width * hrs.height;
            for (int i = 0; i < hrss; i++) {
                hrt[i] = static_cast<float>(hrotation.at<double>(i));
            }
            OMatrix = homographyMatrix;

            zeno::log_info("homographyMatrix.width:{} homographyMatrix.height:{}",hs.width,hs.height);
        }

//essentialMatrix
        if(beM){
            zeno::log_info("beM");
            cv::Mat essentialMatrix, mask;
//            essentialMatrix = cv::findEssentialMat(points1, points2, cameraMatrix, cv::RANSAC, 0.99, 1.0, 1000, noArray());
            essentialMatrix = cv::findEssentialMat(points1, points2, 1.0, Point2d(0, 0),RANSAC,0.999,1.0, 1000,noArray());

            zeno::log_info("essentialMatrix:{}x{}",essentialMatrix.cols,essentialMatrix.rows);
            auto &em = image3->uvs.add_attr<float>("essentialMatrix");
            cv::Size es = essentialMatrix.size();
            int ess = es.width * es.height;

            for (int i = 0; i < ess; i++) {
                em[i] = static_cast<float>(essentialMatrix.at<double>(i));
            }

            cv::Mat rotation,translation;
            cv::recoverPose(essentialMatrix,points1,points2, cameraMatrix, rotation, translation);

            zeno::log_info("recoverPose_essentialMatrix");
            auto &rt = image3->uvs.add_attr<float>("essentialRotation");
            auto &tl = image3->uvs.add_attr<float>("essentialTranslation");

            cv::Size ts = translation.size();
            int tss = ts.width * ts.height;
            for (int i = 0; i < tss; i++) {
                tl[i] = static_cast<float>(translation.at<double>(i));
            }
            cv::Size rs = rotation.size();
            int rss = rs.width * rs.height;
            for (int i = 0; i < rss; i++) {
                rt[i] = static_cast<float>(rotation.at<double>(i));
            }
            OMatrix = essentialMatrix;
            zeno::log_info("essentialMatrix.width:{} essentialMatrix.height:{}",es.width,es.height);
        }

        if(visualize && !stitch){
            int h = h1;
            int w = w1+w2;
            cv::Mat V(h,w,CV_8UC3);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb = image1->verts[i * w1 + j];
                    cv::Vec3b& pixel = V.at<cv::Vec3b>(i, j);
                    pixel[0] = rgb[0] * 255;
                    pixel[1] = rgb[1] * 255;
                    pixel[2] = rgb[2] * 255;
                }
                for (int j = w1; j < w; j++) {
                    vec3f rgb = image2->verts[i * w2 + j - w1];
                    cv::Vec3b& pixel = V.at<cv::Vec3b>(i, j);
                    pixel[0] = rgb[0] * 255;
                    pixel[1] = rgb[1] * 255;
                    pixel[2] = rgb[2] * 255;
                }
            }
            cv::Scalar lineColor(255, 0, 255);
#pragma omp parallel for
            for (size_t i = 0; i < points1.size(); i++) {
                cv::Point2f pt1 = points1[i];
                cv::Point2f pt2 = points2[i] + cv::Point2f(w1, 0);
                cv::line(V, pt1, pt2, lineColor, 2);
            }
            cv::Size vs = V.size();
            image3->verts.resize(vs.width * vs.height);
            ud3.set2("w", vs.width);
            ud3.set2("h", vs.height);
#pragma omp parallel for
            for (int i = 0; i < vs.width * vs.height; i++) {
                cv::Vec3b pixel = V.at<cv::Vec3b>(i);
                image3->verts[i][0] = zeno::min(static_cast<float>(pixel[0])/255,1.0f);
                image3->verts[i][1] = zeno::min(static_cast<float>(pixel[1])/255,1.0f);
                image3->verts[i][2] = zeno::min(static_cast<float>(pixel[2])/255,1.0f);
            }
        }
        if(stitch){
            cv::Mat result;
            cv::Stitcher::Status status = stitcher->stitch(images, result);
            cv::Size vs = result.size();
            image3->verts.resize(vs.width * vs.height);

            if (status == cv::Stitcher::OK) {
                int w = result.cols;
                int h = result.rows;
                image3->verts.resize(w*h);
                ud3.set2("w",w);
                ud3.set2("h",h);
                for (int i = 0; i < h; i++) {
                    for (int j = 0; j < w; j++) {
                        cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                        image3->verts[i * w + j][0] = static_cast<float>(pixel[0])/255;
                        image3->verts[i * w + j][1] = static_cast<float>(pixel[1])/255;
                        image3->verts[i * w + j][2] = static_cast<float>(pixel[2])/255;
                    }
                }
            }
            else {
                zeno::log_info("stitching failed");
            }
        }
//        CVImageObject CVMatrix1(OMatrix);
//        auto CVMatrix = std::make_shared<CVImageObject>(CVMatrix1);
//        set_output("matrix",std::move(CVMatrix));
        set_output("image", image3);
    }
};

ZENDEFNODE(ImageFeatureMatch, {
    {
        { "image1" },
        { "image2" },
        { "float", "maxMatchDistance", "0.7" },
        { "bool", "perspectiveMatrix", "1" },
        { "bool", "fundamentalMatrix", "0" },
        { "bool", "essentialMatrix", "1" },
        { "bool", "homographyMatrix", "0" },
        { "bool", "visualize", "1" },
        { "bool", "stitch", "0" },
    },
    {
//        { "matrix" },
        { "image" },
    },
    {},
    { "image" },
});

Point2f pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2f
    (
        ( p.x - K.at<float>(0,2) ) / K.at<float>(0,0),
        ( p.y - K.at<float>(1,2) ) / K.at<float>(1,1)
    );
}
Point2f cam2pixel(const Point2f& p, const Mat& K) {
    return Point2f(
        p.x * K.at<float>(0, 0) + K.at<float>(0, 2),
        p.y * K.at<float>(1, 1) + K.at<float>(1, 2)
    );
}
struct Image3DAnalyze : INode {
    void apply() override {
        auto image1 = get_input<PrimitiveObject>("image1");
        auto image2 = get_input<PrimitiveObject>("image2");
        auto visualize = get_input2<bool>("visualize");
        auto visualize2 = get_input2<bool>("visualize2");
        auto &ud1 = image1->userData();
        int w1 = ud1.get2<int>("w");
        int h1 = ud1.get2<int>("h");
        auto &ud2 = image2->userData();
        int w2 = ud2.get2<int>("w");
        int h2 = ud2.get2<int>("h");

        auto image3 = std::make_shared<PrimitiveObject>();
        auto &ud3 = image3->userData();
        image3->verts.resize(w2 * h2);
        image3->uvs.resize(zeno::max(12,zeno::max(image1->uvs.size(),image2->uvs.size())));
        ud3.set2("h", h2);
        ud3.set2("w", w2);

        image3->verts = image2->verts;

        cv::Mat imagecvin1(h1, w1, CV_8UC3);
        cv::Mat imagecvin2(h2, w2, CV_8UC3);
        for (int i = 0; i < h1; i++) {
            for (int j = 0; j < w1; j++) {
                vec3f rgb = image1->verts[i * w1 + j];
                cv::Vec3b& pixel = imagecvin1.at<cv::Vec3b>(i, j);
                pixel[0] = rgb[0] * 255;
                pixel[1] = rgb[1] * 255;
                pixel[2] = rgb[2] * 255;
            }
        }
        for (int i = 0; i < h2; i++) {
            for (int j = 0; j < w2; j++) {
                vec3f rgb = image2->verts[i * w2 + j];
                cv::Vec3b& pixel = imagecvin2.at<cv::Vec3b>(i, j);
                pixel[0] = rgb[0] * 255;
                pixel[1] = rgb[1] * 255;
                pixel[2] = rgb[2] * 255;
            }
        }

//cameraMatrix
        float fx = w1;   // image.width;
        float fy = h1;   // image.height;
        float cx = w1/2; // image.width / 2.0;
        float cy = h1/2; // image.height / 2.0;
        cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << fx,0,cx,
                0,fy,cy,
                0, 0, 1);

        auto &cm = image3->uvs.add_attr<float>("cameraMatrix");
        cv::Size cs = cameraMatrix.size();
        int css = cs.width * cs.height;
        for (int i = 0; i < css; i++) {
            cm[i] = static_cast<float>(cameraMatrix.at<float>(i));
        }
//distCoeffsMatrix
        cv::Mat distCoeffsMatrix = (cv::Mat_<float>(4, 1) << 1,1,1,1);

//matchpoints
        std::vector<cv::Point2f> image1Points,image2Points;
        std::vector<cv::Point2f> image1PointsP2C,image2PointsP2C;
        int flag = image1->uvs.size()>=image2->uvs.size()?1:0;
        zeno::log_info("flag:{}",flag);

        auto &mp1 = image3->uvs.add_attr<vec3f>("image1MatchPoints") ;
        mp1 = image2->uvs.add_attr<vec3f>("image1MatchPoints");
        auto &mp2 = image3->uvs.add_attr<vec3f>("image2MatchPoints") ;
        mp2 = image2->uvs.add_attr<vec3f>("image2MatchPoints");
        auto &mp1P2C = image3->uvs.add_attr<vec3f>("image1MatchPointsP2C");
        auto &mp2P2C = image3->uvs.add_attr<vec3f>("image2MatchPointsP2C");
//        image3->uvs.resize(image2->uvs.size());

        for(size_t i = 0;i < image3->uvs.size();i++){
            cv::Point2f pt1(mp1[i][1], mp1[i][2]);
            image1Points.push_back(pt1);
            cv::Point2f pt1P2C = pixel2cam( pt1 , cameraMatrix);
            image1PointsP2C.push_back ( pt1P2C );
            mp1P2C[i] = {mp1[i][0],pt1P2C.x,pt1P2C.y};

            cv::Point2f pt2(mp2[i][1], mp2[i][2]);
            image2Points.push_back(pt2);
            cv::Point2f pt2P2C = pixel2cam( pt2 , cameraMatrix);
            image2PointsP2C.push_back ( pt2P2C );
            mp2P2C[i] = {mp2[i][0],pt2P2C.x,pt2P2C.y};
        }
        zeno::log_info("image1Points.size:{} image2Points.size:{}",image1Points.size(),image2Points.size());

        cv::Mat points1Mat(2,image1Points.size(), CV_32FC1);
        cv::Mat points2Mat(2,image2Points.size(), CV_32FC1);
        for (size_t i = 0; i < image1PointsP2C.size(); i++) {
            points1Mat.at<float>(0, i) = image1PointsP2C[i].x;
            points1Mat.at<float>(1, i) = image1PointsP2C[i].y;
            points2Mat.at<float>(0, i) = image2PointsP2C[i].x;
            points2Mat.at<float>(1, i) = image2PointsP2C[i].y;
        }
        zeno::log_info("points1Mat.size:{} points2Mat.size:{}",points1Mat.cols,points2Mat.cols);

//essentialMatrix1
        auto &em1 = image1->uvs.add_attr<float>("essentialMatrix");
        auto &er1 = image1->uvs.add_attr<float>("essentialRotation");
        auto &et1 = image1->uvs.add_attr<float>("essentialTranslation");
        image3->uvs.add_attr<float>("essentialMatrix1") = image1->uvs.add_attr<float>("essentialMatrix");
        cv::Mat essentialMatrix1 = (cv::Mat_<float>(3, 3) << em1[0],em1[1],em1[2],
                                                              em1[3],em1[4],em1[5],
                                                              em1[6],em1[7],em1[8]);

        cv::Mat erotation1 = (cv::Mat_<float>(3, 3) << er1[0],er1[1],er1[2],
                                                        er1[3],er1[4],er1[5],
                                                        er1[6],er1[7],er1[8]);

        cv::Mat etranslation1 = (cv::Mat_<float>(3, 1) << et1[0],et1[1],et1[2]);

        cv::Size ems1 = essentialMatrix1.size();
        zeno::log_info("essentialMatrix1.width:{} essentialMatrix1.height:{}",ems1.width,ems1.height);

//essentialMatrix2
        auto &em2 = image2->uvs.add_attr<float>("essentialMatrix");
        auto &er2 = image2->uvs.add_attr<float>("essentialRotation");
        auto &et2 = image2->uvs.add_attr<float>("essentialTranslation");
        image3->uvs.add_attr<float>("essentialMatrix2") = image2->uvs.add_attr<float>("essentialMatrix");
        cv::Mat essentialMatrix2 = (cv::Mat_<float>(3, 3) << em2[0],em2[1],em2[2],
                                                              em2[3],em2[4],em2[5],
                                                              em2[6],em2[7],em2[8]);

        cv::Mat erotation2 = (cv::Mat_<float>(3, 3) << er2[0],er2[1],er2[2],
                                                        er2[3],er2[4],er2[5],
                                                        er2[6],er2[7],er2[8]);

        cv::Mat etranslation2 = (cv::Mat_<float>(3, 1) << et2[0],et2[1],et2[2]);

        cv::Size ems2 = essentialMatrix2.size();
        zeno::log_info("essentialMatrix2.width:{} essentialMatrix2.height:{}",ems2.width,ems2.height);

        cv::Mat cameraMatrix1 = cv::Mat::ones(3, 4, CV_32F);
        cv::Size cmp1si = cameraMatrix1.size();
        cv::Mat cameraMatrix2 = cv::Mat::ones(3, 4, CV_32F);
        cv::Size imageSize(w1, h1);

// estimateCameraMatrix1
        auto &cm1 = image3->uvs.add_attr<float>("cameraMatrix1");
        cv::Mat K1 = cameraMatrix;
        cv::Mat distCoeffs1 = distCoeffsMatrix;
        cv::Mat R1 = erotation1;
        cv::Mat t1 = etranslation1;
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K1, distCoeffs1, imageSize , R1, cameraMatrix1);
        cv::Size cmp1s = cameraMatrix1.size();
        zeno::log_info("cameraMatrix1.width:{} cameraMatrix1.height:{}",cmp1s.width,cmp1s.height);
        for (int i = 0; i < 9; i++) {
            cm1[i] = static_cast<float>(cameraMatrix1.at<float>(i));
        }
// estimateCameraMatrix2
        auto &cm2 = image3->uvs.add_attr<float>("cameraMatrix2");
        cv::Mat K2 = cameraMatrix;
        cv::Mat distCoeffs2 = distCoeffsMatrix;
        cv::Mat R2 = erotation2;
        cv::Mat t2 = etranslation2;
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K2, distCoeffs2, imageSize , R2, cameraMatrix2);
        cv::Size cmp2s = cameraMatrix2.size();
        zeno::log_info("cameraMatrix2.width:{} cameraMatrix2.height:{}",cmp2s.width,cmp2s.height);
        for (int i = 0; i < 9; i++) {
            cm2[i] = static_cast<float>(cameraMatrix2.at<float>(i));
        }
//3x3 -> 3x4
        auto &cmp1 = image3->uvs.add_attr<float>("cameraMatrixPro1");
        auto &cmp2 = image3->uvs.add_attr<float>("cameraMatrixPro2");
        cv::Mat cameraMatrixPro1x = cv::Mat::eye(3, 4, CV_32F);
        cv::Mat roi1 = cameraMatrixPro1x(cv::Rect(0, 0, 3, 3));
        cameraMatrix1.copyTo(roi1);
        cv::Mat cameraMatrixPro2x = cv::Mat::eye(3, 4, CV_32F);
        cv::Mat roi2 = cameraMatrixPro2x(cv::Rect(0, 0, 3, 3));
        cameraMatrix2.copyTo(roi2);
        for (int i = 0; i < 12; i++) {
            cmp1[i] = static_cast<float>(cameraMatrixPro1x.at<float>(i));
        }
        for (int i = 0; i < 12; i++) {
            cmp2[i] = static_cast<float>(cameraMatrixPro2x.at<float>(i));
        }
        zeno::log_info("cameraMatrixPro1.width:{} cameraMatrixPro1.height:{}",cameraMatrixPro1x.cols,cameraMatrixPro1x.rows);
        zeno::log_info("cameraMatrixPro2.width:{} cameraMatrixPro2.height:{}",cameraMatrixPro2x.cols,cameraMatrixPro2x.rows);
//todo:

//4D points
// triangulatePoints init
        auto &p4d = image3->uvs.add_attr<vec4f>("4DPoints");
        auto &p3d = image3->uvs.add_attr<vec3f>("3DPoints");
        cv::Mat points4D = cv::Mat::ones(3, 4, CV_32F);

// triangulatePoints method1

//        Mat T1 = (Mat_<float> (3,4) <<
//            1,0,0,0,
//            0,1,0,0,
//            0,0,1,0);
        cv::Mat T1 = (cv::Mat_<float> (3,4) <<
                erotation1.at<float>(0,0), erotation1.at<float>(0,1), erotation1.at<float>(0,2), etranslation1.at<float>(0,0),
                erotation1.at<float>(1,0), erotation1.at<float>(1,1), erotation1.at<float>(1,2), etranslation1.at<float>(1,0),
                erotation1.at<float>(2,0), erotation1.at<float>(2,1), erotation1.at<float>(2,2), etranslation1.at<float>(2,0)
        );
        cv::Mat T2 = (cv::Mat_<float> (3,4) <<
                erotation2.at<float>(0,0), erotation2.at<float>(0,1), erotation2.at<float>(0,2), etranslation2.at<float>(0,0),
                erotation2.at<float>(1,0), erotation2.at<float>(1,1), erotation2.at<float>(1,2), etranslation2.at<float>(1,0),
                erotation2.at<float>(2,0), erotation2.at<float>(2,1), erotation2.at<float>(2,2), etranslation2.at<float>(2,0)
        );
        //相机内外参数整合为一个相机投影矩阵
//        cv::Mat C1,C2;
        cv::Mat C1 = cv::Mat::zeros(3, 4, CV_32FC1);
        cv::Mat C2 = cv::Mat::zeros(3, 4, CV_32FC1);
        cv::gemm(cameraMatrix,T1, 1.0, cv::Mat(), 0.0, C1);
        cv::gemm(cameraMatrix,T2, 1.0, cv::Mat(), 0.0, C2);
        zeno::log_info("C1.cols:{},C1.rows:{}",C1.cols,C1.rows);
        cv::triangulatePoints(T1,T2,points1Mat, points2Mat, points4D);

// triangulatePoints method2
//        cv::triangulatePoints(cameraMatrixPro1x, cameraMatrixPro2x, points1Mat, points2Mat, points4D);
        zeno::log_info("points4D.cols:{},points4D.rows:{}",points4D.cols,points4D.rows);
        for (size_t j = 0; j < points4D.cols; j++) {
            for(size_t i = 0;i < points4D.rows;i++){
                p4d[j][i] = points4D.at<float>(i,j);
            }
        }

//        cv::triangulatePoints(cameraMatrixPro,cameraMatrixPro, points1, points2, points4D);
//        for (size_t i = 0; i < points4D.rows; i++) {
//            cv::Point4f point = points4D.at<cv::Point4f>(i);
//            p4d[i] = {point.x,point.y,point.z,point.w};
//        }
        int numPoints = points4D.cols; // 获取点的数量
        image3->verts.resize(numPoints);
        float maxdepth = 0;
        float mindepth = 0;
        for (int i = 0; i < numPoints; i++) {
            cv::Vec4f point = points4D.col(i);
            float x = point(0);
            float y = point(1);
            float z = point(2);
            float w = point(3);
            p4d[i] = {x,y,z,w};
            cv::Point2f pt(x/w,y/w);
            cam2pixel(pt,cameraMatrix);
            p3d[i] = {image2Points[i].x,image2Points[i].y,z/w};
            maxdepth = zeno::max(maxdepth,z/w);
            mindepth = zeno::min(mindepth,z/w);
            image3->verts[i] = p3d[i];
        }
        zeno::log_info("triangulatePoints");
//        ud3.set2("isImage", 0);
        if(visualize2){
            ud3.set2("isImage", 1);
            image3->verts.resize(0);
            image3->verts.resize(w2 * h2);
            for(size_t i = 0;i < image3->uvs.size();i++){
                float var = (float)(p3d[i][2]/(maxdepth - mindepth));
                int idx = (int)p3d[i][0] * w2 + (int)p3d[i][1];
                image3->verts[idx] = {var,var,var};
            }
        }
        if(visualize){
            ud3.set2("isImage", 1);
            int h = h1;
            int w = w1+w2;
            cv::Mat V(h,w,CV_8UC3);
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w1; j++) {
                    vec3f rgb = image1->verts[i * w1 + j];
                    cv::Vec3b& pixel = V.at<cv::Vec3b>(i, j);
                    pixel[0] = rgb[0] * 255;
                    pixel[1] = rgb[1] * 255;
                    pixel[2] = rgb[2] * 255;
                }
                for (int j = w1; j < w; j++) {
                    vec3f rgb = image2->verts[i * w2 + j - w1];
                    cv::Vec3b& pixel = V.at<cv::Vec3b>(i, j);
                    pixel[0] = rgb[0] * 255;
                    pixel[1] = rgb[1] * 255;
                    pixel[2] = rgb[2] * 255;
                }
            }
            cv::Scalar lineColor(0, 255, 255);
#pragma omp parallel for
            for (size_t i = 0; i < image1Points.size(); i++) {
                cv::Point2f pt1 = image1Points[i];
                cv::Point2f pt2 = image2Points[i] + cv::Point2f(w1, 0);
                cv::line(V, pt1, pt2, lineColor, 2);
            }
            cv::Size vs = V.size();
            image3->verts.resize(vs.width * vs.height);
            ud3.set2("w", vs.width);
            ud3.set2("h", vs.height);
#pragma omp parallel for
            for (int i = 0; i < vs.width * vs.height; i++) {
                cv::Vec3b pixel = V.at<cv::Vec3b>(i);
                image3->verts[i][0] = zeno::min(static_cast<float>(pixel[0])/255,1.0f);
                image3->verts[i][1] = zeno::min(static_cast<float>(pixel[1])/255,1.0f);
                image3->verts[i][2] = zeno::min(static_cast<float>(pixel[2])/255,1.0f);
            }
        }
        set_output("image", image3);
    }
};

ZENDEFNODE(Image3DAnalyze, {
    {
        { "image1" },
        { "image2" },
        { "bool", "visualize", "0" },
        { "bool", "visualize2", "1" },
    },
    {
        { "image" },
    },
    {},
    { "image" },
});

struct CreateCameraMatrix : INode {
    void apply() override {
        auto top = get_input2<vec3f>("top");
        auto mid = get_input2<vec3f>("mid");
        auto bot = get_input2<vec3f>("bot");

        cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_32F);
        cameraMatrix.at<float>(0, 0) = top[0];
        cameraMatrix.at<float>(1, 1) = mid[1];
        cameraMatrix.at<float>(0, 2) = top[2];
        cameraMatrix.at<float>(1, 2) = mid[2];

//        cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << top[0],top[1],top[2],
//                                                       mid[0],mid[1],mid[2],
//                                                       bot[0],bot[1],bot[2]);

//        cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << fx,0,cx,
//                                                         0,fy,cy,
//                                                         0, 0, 1);

        CVImageObject CVMatrix1(cameraMatrix);
        auto CVMatrix = std::make_shared<CVImageObject>(CVMatrix1);
        set_output("matrix",std::move(CVMatrix));
    }
};

ZENDEFNODE(CreateCameraMatrix, {
    {
        {"vec3f", "top", "1000,0,640"},
        {"vec3f", "mid", "0,800,480"},
        {"vec3f", "bot", "0,0,1"},
    },
    {
        { "matrix" },
    },
    {},
    { "image" },
});

struct EstimateCameraMatrix : INode {
    void apply() override {
        auto image1 = get_input<PrimitiveObject>("image1");
        auto image2 = get_input<PrimitiveObject>("image2");
        auto depth = get_input2<float>("depth");
        auto &ud1 = image1->userData();
        int w1 = ud1.get2<int>("w");
        int h1 = ud1.get2<int>("h");
        auto &ud2 = image2->userData();
        int w2 = ud2.get2<int>("w");
        int h2 = ud2.get2<int>("h");

        cv::Mat imagecvin1(h1, w1, CV_8UC3);
        cv::Mat imagecvin2(h2, w2, CV_8UC3);
        for (int i = 0; i < h1; i++) {
            for (int j = 0; j < w1; j++) {
                vec3f rgb1 = image1->verts[i * w1 + j];
                cv::Vec3b& pixel = imagecvin1.at<cv::Vec3b>(i, j);
                pixel[0] = rgb1[0] * 255;
                pixel[1] = rgb1[1] * 255;
                pixel[2] = rgb1[2] * 255;
            }
        }
        for (int i = 0; i < h2; i++) {
            for (int j = 0; j < w2; j++) {
                vec3f rgb2 = image2->verts[i * w2 + j];
                cv::Vec3b& pixel = imagecvin2.at<cv::Vec3b>(i, j);
                pixel[0] = rgb2[0] * 255;
                pixel[1] = rgb2[1] * 255;
                pixel[2] = rgb2[2] * 255;
            }
        }

        std::vector<cv::Point2f> image1Points,image2Points;
        std::vector<cv::Point3f> objectPoints;

        int flag = image1->uvs.size()>=image2->uvs.size()?1:0;
        zeno::log_info("flag:{}",flag);
        auto &m11 = image1->uvs.add_attr<vec3f>("image1MatchPoints");
        auto &m21 = image2->uvs.add_attr<vec3f>("image1MatchPoints");
        auto &m12 = image1->uvs.add_attr<vec3f>("image2MatchPoints");
        auto &m22 = image2->uvs.add_attr<vec3f>("image2MatchPoints");
        std::set<float> dp;
        std::set<float> dp1;
        std::set<float> dp2;

        if(flag == 1){
            for(size_t i = 0;i < image2->uvs.size();i++){
                dp1.insert(m12[i][0]);
            }
            for(size_t i = 0;i < image1->uvs.size();i++){
                dp1.insert(m22[i][0]);
                if (dp1.count(m22[i][0]) == 2 ) {
                    cv::Point2f pt1(m12[i][1], m12[i][2]);
                    cv::Point3f pt(m11[i][1], m11[i][2],depth);
                    image1Points.push_back(pt1);
                    objectPoints.push_back(pt);
                }
            }
            for(size_t i = 0;i < image1->uvs.size();i++){
                dp2.insert(m11[i][0]);
            }
            for(size_t i = 0;i < image2->uvs.size();i++){
                dp2.insert(m21[i][0]);
                if (dp2.count(m21[i][0]) == 2 ) {
                    cv::Point2f pt2(m22[i][1], m22[i][2]);
                    image2Points.push_back(pt2);
                }
            }
        }
        if(flag == 0){
            for(size_t i = 0;i < image2->uvs.size();i++){
                dp1.insert(m12[i][0]);
            }
            for(size_t i = 0;i < image1->uvs.size();i++){
                dp1.insert(m22[i][0]);
                if (dp1.count(m22[i][0]) == 2 ) {
                    cv::Point2f pt1(m12[i][1], m12[i][2]);
                    image1Points.push_back(pt1);
                }
            }
            for(size_t i = 0;i < image1->uvs.size();i++){
                dp2.insert(m11[i][0]);
            }
            for(size_t i = 0;i < image2->uvs.size();i++){
                dp2.insert(m21[i][0]);
                if (dp2.count(m21[i][0]) == 2 ) {
                    cv::Point2f pt2(m22[i][1], m22[i][2]);
                    cv::Point3f pt(m21[i][1], m21[i][2],depth);
                    image2Points.push_back(pt2);
                    objectPoints.push_back(pt);
                }
            }
        }

        cv::Mat cameraMatrix1,cameraMatrix2;
        cv::Mat distCoeffs1,distCoeffs2;
        cv::Mat R,T;
        cv::Mat rvecs, tvecs;
        int flags = fisheye::CALIB_FIX_INTRINSIC;
        TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6);
//        double rms = cv::stereoCalibrate(
//                objectPoints, imagePoints1, imagePoints2,
//                cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
//                cv::Size(640, 480), // 图像的大小
//                cv::Mat(), cv::Mat(), // 旋转矩阵和平移向量（输出参数）
//                cv::CALIB_FIX_INTRINSIC | cv::CALIB_USE_INTRINSIC_GUESS // 标定标志
//        );
        float rms = stereoCalibrate(objectPoints, image1Points, image2Points,
                                     cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                                     cv::Size(w1, h1),
                                     R, T,
                                     rvecs, tvecs,
                                     flags,
                                     criteria);
        zeno::log_info("rms:{}",rms);

        auto &cm2 = image2->uvs.add_attr<float>("cameraMatrix");
        cv::Size cs2 = cameraMatrix2.size();
        int css2 = cs2.width * cs2.height;
        for (int i = 0; i < css2; i++) {
            cm2[i] = static_cast<float>(cameraMatrix2.at<float>(i));
        }
        set_output("image", image2);
    }
};

ZENDEFNODE(EstimateCameraMatrix, {
    {
        {"image1" },
        {"image2" },
        { "float", "depth", "0" },
    },
    {
        { "image" },
    },
    {},
    { "deprecated" },
});

}
}