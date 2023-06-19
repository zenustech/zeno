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
static void
normalMap(std::shared_ptr<PrimitiveObject> &grayImage, int width, int height, std::vector<float> &normal) {
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

static void
scharr2(std::shared_ptr<PrimitiveObject> &src, std::shared_ptr<PrimitiveObject> &dst, int width, int height,
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

            // Calculate gradient magnitude
            int mag = std::sqrt(gx[idx] * gx[idx] + gy[idx] * gy[idx]);
            // Apply threshold
            if (mag * 255 > threshold) {
                // Set to white
                dst->verts[idx] = {1, 1, 1};
            } else {
                // Set to black
                dst->verts[idx] = {0, 0, 0};
            }
            // Clamp to [0, 255] and store in output image
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
            // 自适应阈值化
            cv::Mat thresholdImage;
            int maxValue = 255;  // 最大像素值
            int blockSize = 3;  // 邻域块大小
            double C = 2;  // 常数项
            cv::adaptiveThreshold(imagecvin, thresholdImage, maxValue, cv::ADAPTIVE_THRESH_MEAN_C,
                                  cv::THRESH_BINARY, blockSize, C);

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

//边缘检测
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
            // 自适应阈值化
            cv::Mat thresholdImage;
            int maxValue = 255;  // 最大像素值
            int blockSize = 3;  // 邻域块大小
            double C = 2;  // 常数项
            cv::adaptiveThreshold(imagecvin, thresholdImage, maxValue, cv::ADAPTIVE_THRESH_MEAN_C,
                                  cv::THRESH_BINARY, blockSize, C);

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
        auto &dp = image->tris.add_attr<float>("descriptors"); //descriptors.size() = 32 * nFeatures

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
        image->tris.resize(dss);
        zeno::log_info("orbDescriptor.width:{}, orbDescriptor.height:{}",ds.width, ds.height);
        for (int i = 0; i < dss; i++) {
            dp[i] = imageFloat.at<float>(i);
        }
        for (const cv::KeyPoint &keypoint: ikeypoints) {
            float x = static_cast<float>(keypoint.pt.x);
            float y = static_cast<float>(keypoint.pt.y);
            image->uvs.push_back({x, y});
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
        auto &dp = image->tris.add_attr<float>("descriptors");
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
        image->tris.resize(dss);
        zeno::log_info("siftDescriptor.width:{}, siftDescriptor.height:{}",ds.width, ds.height);
        for (int i = 0; i < dss; i++) {
            dp[i] = imageFloat.at<float>(i);
        }
        for (const cv::KeyPoint &keypoint: ikeypoints) {
            float x = static_cast<float>(keypoint.pt.x);
            float y = static_cast<float>(keypoint.pt.y);
            image->uvs.push_back({x, y});
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
        { "float", "sigma", "1.6" }, //高斯滤波器的初始尺度
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

        auto &ud1 = image1->userData();
        int w1 = ud1.get2<int>("w");
        int h1 = ud1.get2<int>("h");
        auto &ud2 = image2->userData();
        int w2 = ud2.get2<int>("w");
        int h2 = ud2.get2<int>("h");

        auto image3 = std::make_shared<PrimitiveObject>();
        auto &ud3 = image3->userData();
        image3->verts.resize(w2 * h2);
        image3->tris.resize(zeno::max(image1->tris.size(),image2->tris.size()));
        ud3.set2("h", h2);
        ud3.set2("w", w2);
        ud3.set2("isImage", 1);
        image3->verts = image2->verts;

        std::vector<cv::KeyPoint> keypoints;
        auto d1 = image1->tris.add_attr<float>("descriptors");
        auto d2 = image2->tris.add_attr<float>("descriptors");
        auto k1 = image1->uvs;
        auto k2 = image2->uvs;
        int ks1 = k1.size();
        int ks2 = k2.size();
        int dw1 = d1.size()/ks1;
        int dw2 = d2.size()/ks2;

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
        auto &md = image3->tris.add_attr<float>("matchDistance");
        int mdi = 0;
        for (const auto& knnMatch : knnMatches) {
            if (knnMatch.size() < 2) {
                continue;
            }
            float distanceRatio = knnMatch[0].distance / knnMatch[1].distance;
            if (distanceRatio < matchD) {
                filteredMatches.push_back(knnMatch[0]);
                md[mdi] = static_cast<float>(knnMatch[0].distance);
                mdi++;
            }
        }
        zeno::log_info("knnMatches.size:{},filteredMatches.size：{}",knnMatches.size(),filteredMatches.size());

        std::vector<cv::Point2f> points1, points2;
        auto &m1 = image3->tris.add_attr<vec3f>("image1MatchPoints");
        auto &m2 = image3->tris.add_attr<vec3f>("image2MatchPoints");
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

//cameraMatrix
        float fx = w1;   // image.width;
        float fy = h1;   // image.height;
        float cx = w1/2; // image.width / 2.0;
        float cy = h1/2; // image.height / 2.0;
        cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << fx,0,cx,
        0,fy,cy,
        0, 0, 1);

        if(bfM || beM || bhM){
            auto &cm = image3->tris.add_attr<float>("cameraMatrix");
            cv::Size cs = cameraMatrix.size();
            int css = cs.width * cs.height;
            for (int i = 0; i < css; i++) {
                cm[i] = static_cast<float>(cameraMatrix.at<float>(i));
            }
        }

//perspectiveMatrix
        if(bpM){
            auto &pp1 = image3->tris.add_attr<vec3f>("perspectiveImage1Points");
            auto &pp2 = image3->tris.add_attr<vec3f>("perspectiveImage2Points");
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

            auto &pm = image3->tris.add_attr<float>("perspectiveMatrix");
            cv::Size ps = perspectiveMatrix.size();
            int pss = ps.width * ps.height;
            for (int i = 0; i < pss; i++) {
                pm[i] = static_cast<float>(perspectiveMatrix.at<double>(i));
            }
            auto &ptl = image3->tris.add_attr<float>("perspectiveTranslation");
            auto &prt = image3->tris.add_attr<float>("perspectiveRotation");

            ptl[0] = pm[2];
            ptl[1] = pm[5];
            ptl[2] = pm[8];

            prt[0] = pm[0];
            prt[1] = pm[1];
            prt[2] = pm[3];
            prt[3] = pm[4];

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
            auto &fm = image3->tris.add_attr<float>("fundamentalMatrix");
            cv::Size fs = fundamentalMatrix.size();
            int fss = fs.width * fs.height;
            for (int i = 0; i < fss; i++) {
                fm[i] = static_cast<float>(fundamentalMatrix.at<double>(i));
            }
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
            auto &hm = image3->tris.add_attr<float>("homographyMatrix");
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
            auto &hrt = image3->tris.add_attr<float>("homographyRotation");
            auto &htl = image3->tris.add_attr<float>("homographyTranslation");
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
            zeno::log_info("homographyMatrix.width:{} homographyMatrix.height:{}",hs.width,hs.height);
        }


//essentialMatrix
        if(beM){
            cv::Mat essentialMatrix, mask;
            double threshold = 1.0;
            double prob = 0.99;
            int method = cv::RANSAC;
            essentialMatrix = cv::findEssentialMat(points1, points2, cameraMatrix, method, prob, threshold, mask);

            auto &em = image3->tris.add_attr<float>("essentialMatrix");
            cv::Size es = essentialMatrix.size();
            int ess = es.width * es.height;
            for (int i = 0; i < ess; i++) {
                em[i] = static_cast<float>(essentialMatrix.at<double>(i));
            }

            cv::Mat rotation,translation;
            cv::recoverPose(essentialMatrix,points1,points2, cameraMatrix, rotation, translation);

            auto &rt = image3->tris.add_attr<float>("essentialRotation");
            auto &tl = image3->tris.add_attr<float>("essentialTranslation");

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
        set_output("image", image3);
    }
};

ZENDEFNODE(ImageFeatureMatch, {
    {
        { "image1" },
        { "image2" },
        {"float", "maxMatchDistance", "0.7"},
        { "bool", "perspectiveMatrix", "1" },
        { "bool", "fundamentalMatrix", "0" },
        { "bool", "essentialMatrix", "0" },
        { "bool", "homographyMatrix", "0" },
        { "bool", "visualize", "1" },
        { "bool", "stitch", "0" },
    },
    {
        { "image" },
    },
    {},
    { "image" },
});


//
//struct CameraEstimate : INode {
//    void apply() override {
//        auto image1 = get_input<PrimitiveObject>("image1");
//        auto image2 = get_input<PrimitiveObject>("image2");
//        auto mode = get_input2<std::string>("mode");
//        auto matchD = get_input2<float>("maxMatchDistance");
//        auto visualize = get_input2<bool>("visualize");
//        auto stitch = get_input2<bool>("stitch");
//        auto &ud1 = image1->userData();
//        int w1 = ud1.get2<int>("w");
//        int h1 = ud1.get2<int>("h");
//        auto &ud2 = image2->userData();
//        int w2 = ud2.get2<int>("w");
//        int h2 = ud2.get2<int>("h");
//
//        auto image3 = std::make_shared<PrimitiveObject>();
//        auto &ud3 = image3->userData();
//        image3->verts.resize(w2 * h2);
//        image3->tris.resize(zeno::max(image1->tris.size(),image2->tris.size()));
//        ud3.set2("h", h2);
//        ud3.set2("w", w2);
//        ud3.set2("isImage", 1);
//        image3->verts = image2->verts;
////        ****************
//        cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create();
////        *****************
//        cv::Mat imagecvin1(h1, w1, CV_8UC3);
//        cv::Mat imagecvin2(h2, w2, CV_8UC3);
//        for (int i = 0; i < h1; i++) {
//            for (int j = 0; j < w1; j++) {
//                vec3f rgb = image1->verts[i * w1 + j];
//                cv::Vec3b& pixel = imagecvin1.at<cv::Vec3b>(i, j);
//                pixel[0] = rgb[0] * 255;
//                pixel[1] = rgb[1] * 255;
//                pixel[2] = rgb[2] * 255;
//            }
//        }
//        for (int i = 0; i < h2; i++) {
//            for (int j = 0; j < w2; j++) {
//                vec3f rgb = image2->verts[i * w2 + j];
//                cv::Vec3b& pixel = imagecvin2.at<cv::Vec3b>(i, j);
//                pixel[0] = rgb[0] * 255;
//                pixel[1] = rgb[1] * 255;
//                pixel[2] = rgb[2] * 255;
//            }
//        }
//        std::vector<cv::Mat> images;
//        images.push_back(imagecvin1);
//        images.push_back(imagecvin2);
//
//        std::vector<cv::KeyPoint> keypoints;
//        auto d1 = image1->tris.add_attr<float>("descriptors");
//        auto d2 = image2->tris.add_attr<float>("descriptors");
//        auto k1 = image1->uvs;
//        auto k2 = image2->uvs;
//        int ks1 = k1.size();
//        int ks2 = k2.size();
//        int dw1 = d1.size()/ks1;
//        int dw2 = d2.size()/ks2;
//        zeno::log_info("ks1:{},ks2:{},dw1:{},dw2:{}",ks1,ks2,dw1,dw2);
//
//        cv::Mat imagecvdescriptors1(ks1, dw1, CV_32F);
//        cv::Mat imagecvdescriptors2(ks2, dw2, CV_32F);
//        for (int i = 0; i < ks1; i++) {
//            for (int j = 0; j < dw1; j++) {
//                imagecvdescriptors1.at<float>(i, j) = d1[i * dw1 + j];
//            }
//        }
//        for (int i = 0; i < ks2; i++) {
//            for (int j = 0; j < dw2; j++) {
//                imagecvdescriptors2.at<float>(i, j) = d2[i * dw2 + j];
//            }
//        }
//
//        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
//        std::vector<std::vector<cv::DMatch>> knnMatches;
//        std::vector<cv::DMatch> filteredMatches;
//        matcher->knnMatch(imagecvdescriptors1, imagecvdescriptors2, knnMatches, 2);
//        filteredMatches.reserve(knnMatches.size());
//        auto &md = image3->tris.add_attr<float>("matchDistance");
//        int mdi = 0;
//        for (const auto& knnMatch : knnMatches) {
//            if (knnMatch.size() < 2) {
//                continue;
//            }
//            float distanceRatio = knnMatch[0].distance / knnMatch[1].distance;
//            if (distanceRatio < matchD) {
//                filteredMatches.push_back(knnMatch[0]);
//                md[mdi] = static_cast<float>(knnMatch[0].distance);
//                mdi++;
//            }
//        }
//        zeno::log_info("BRUTEFORCE  knnMatches.size:{},filteredMatches.size：{}",knnMatches.size(),filteredMatches.size());
//
//        std::vector<cv::Point2f> points1, points2;
//        auto &m1 = image3->tris.add_attr<vec3f>("image1matchidx");
//        auto &m2 = image3->tris.add_attr<vec3f>("image2matchidx");
//        int m = 0;
//        for (const auto &match: filteredMatches) {
//            points1.push_back({k1[match.queryIdx][0], k1[match.queryIdx][1]});
//            m1[m] = {static_cast<float>(match.queryIdx),k1[match.queryIdx][0], k1[match.queryIdx][1]};
//            points2.push_back({k2[match.trainIdx][0], k2[match.trainIdx][1]});
//            m2[m] = {static_cast<float>(match.trainIdx),k2[match.trainIdx][0], k2[match.trainIdx][1]};
//            m++;
//        }
////Fisheye
//        //cameraMatrix
//        cv::Size image_size = images[0].size();  // 假设所有图像的尺寸相同
////        cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
//        double fx = image_size.width;
//        double fy = image_size.height;
//        double cx = image_size.width / 2.0;
//        double cy = image_size.height / 2.0;
//        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0,cx,
//                                                           0,fy,cy,
//                                                           0, 0, 1);
//
//        // distCoeffs
////        cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_32F);  // 初始化为0向量
//
////Method1
////        cv::Mat panorama;
////        cv::Mat t;
////        cv::Stitcher::Status status = stitcher->estimateTransform(images, panorama);
////        if (status == cv::Stitcher::OK) {
////            t = panorama.t();
////        } else {
////            zeno::log_info("Stitcher Error");
////        }
////        auto &st = image3->tris.add_attr<float>("stitcheresimatetranslation");
////        cv::Size trs = t.size();
////        for(int i = 0;i < trs.height * trs.width;i++){
////            st[i] = static_cast<float>(t.at<float>(i));
////        }
//
////Method2
////        cv::Mat panorama;
////        cv::Stitcher::Status status = stitcher->stitch(images, panorama);
////        cv::Stitcher::Status status1 = stitcher->composePanorama(images, panorama);
//////        cv::Stitcher::CameraParams cameraParams;
////        if (status1 == cv::Stitcher::OK) {
////            cv::Mat K, R, t;
////            cv::Mat cameraMatrix, distCoeffs;
////            stitcher->estimateCameraParams(cameraMatrix, distCoeffs);
//////            cv::Stitcher::CameraParams params = stitcher.getCameraParams();
////            panorama->getCameraParams(K, R, t); //Intrinsic Matrix,Rotation Matrix,Translation Vector
////        } else {
////            zeno::log_info("Camera parameter estimation failed with status: ",status1);
////        }
////
////solvePnP
//// 特征点的图像坐标 points1, points2
//
//        //EssentialMat
//        cv::Mat essentialMatrix = cv::findEssentialMat(points1, points2, cameraMatrix, cv::RANSAC, 0.999, 1.0);
//
//        cv::Mat rotation, translation;
//        cv::recoverPose(essentialMatrix, points1, points2, cameraMatrix, rotation, translation);
//
//        auto &rt = image3->tris.add_attr<float>("rotation");
//        auto &tl = image3->tris.add_attr<float>("translation");
//
//        cv::Size ts = translation.size();
//        int tss = ts.width * ts.height;
//        for (int i = 0; i < tss; i++) {
//            tl[i] = static_cast<float>(translation.at<float>(i));
//        }
//        cv::Size rs = rotation.size();
//        int rss = rs.width * rs.height;
//        for (int i = 0; i < rss; i++) {
//            rt[i] = static_cast<float>(rotation.at<float>(i));
//        }
//
//        if(visualize && !stitch){
//            int h = h1;
//            int w = w1+w2;
//            cv::Mat V(h,w,CV_8UC3);
//            for (int i = 0; i < h; i++) {
//                for (int j = 0; j < w1; j++) {
//                    vec3f rgb = image1->verts[i * w1 + j];
//                    cv::Vec3b& pixel = V.at<cv::Vec3b>(i, j);
//                    pixel[0] = rgb[0] * 255;
//                    pixel[1] = rgb[1] * 255;
//                    pixel[2] = rgb[2] * 255;
//                }
//                for (int j = w1; j < w; j++) {
//                    vec3f rgb = image2->verts[i * w2 + j - w1];
//                    cv::Vec3b& pixel = V.at<cv::Vec3b>(i, j);
//                    pixel[0] = rgb[0] * 255;
//                    pixel[1] = rgb[1] * 255;
//                    pixel[2] = rgb[2] * 255;
//                }
//            }
//            zeno::log_info("V ok");
//            cv::Scalar lineColor(255, 0, 255);
//#pragma omp parallel for
//            for (size_t i = 0; i < points1.size(); i++) {
//                cv::Point2f pt1 = points1[i];
//                cv::Point2f pt2 = points2[i] + cv::Point2f(w1, 0);
////                cv::Point2f pt2 = points2[i] + cv::Point2f(imagecvin1.cols, 0);
//                cv::line(V, pt1, pt2, lineColor, 2);
//            }
//            cv::Size vs = V.size();
//            image3->verts.resize(vs.width * vs.height);
//            ud3.set2("w", vs.width);
//            ud3.set2("h", vs.height);
//#pragma omp parallel for
//            for (int i = 0; i < vs.width * vs.height; i++) {
//                cv::Vec3b pixel = V.at<cv::Vec3b>(i);
//                image3->verts[i][0] = zeno::min(static_cast<float>(pixel[0])/255,1.0f);
//                image3->verts[i][1] = zeno::min(static_cast<float>(pixel[1])/255,1.0f);
//                image3->verts[i][2] = zeno::min(static_cast<float>(pixel[2])/255,1.0f);
//            }
//        }
//        if(stitch){
////            cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create();
//            cv::Mat result;
//            cv::Stitcher::Status status = stitcher->stitch(images, result);
////            zeno::log_info("status = stitcher->stitch(images, result)");
//            cv::Size vs = result.size();
//            image3->verts.resize(vs.width * vs.height);
//            if (status == cv::Stitcher::OK) {
//                int w = result.cols;
//                int h = result.rows;
//                image3->verts.resize(w*h);
//                ud3.set2("w",w);
//                ud3.set2("h",h);
//                for (int i = 0; i < h; i++) {
//                    for (int j = 0; j < w; j++) {
//                        cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
//                        image3->verts[i * w + j][0] = static_cast<float>(pixel[0])/255;
//                        image3->verts[i * w + j][1] = static_cast<float>(pixel[1])/255;
//                        image3->verts[i * w + j][2] = static_cast<float>(pixel[2])/255;
//                    }
//                }
//            }
//            else {
//                zeno::log_info("stitching failed");
//            }
//        }
//        set_output("image", image3);
//    }
//};
//
//ZENDEFNODE(CameraEstimate, {
//    {
//        { "image1" },
//        { "image2" },
//        {"enum BRUTEFORCE HAMMING", "mode", "BRUTEFORCE"},
//        {"float", "maxMatchDistance", "0.7"},
//        { "bool", "visualize", "0" },
//        { "bool", "stitch", "0" },
////        { "bool", "stitch2", "0" },
//    },
//    {
//        { "image" },
//    },
//    {},
//    { "image" },
//});
//



struct ImageStitching : INode {
    void apply() override {
//        auto primList = get_input<ListObject>("listPrim")->getRaw<PrimitiveObject>();
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
}
}