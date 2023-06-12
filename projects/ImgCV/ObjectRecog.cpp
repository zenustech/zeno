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
//#include <opencv2/xfeatures2d.hpp>
#include <opencv2/stitching.hpp>
#include <zeno/types/ListObject.h>

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
        auto &dp = image->verts.add_attr<float>("descriptors");
        auto &pos = image->verts.attr<vec3f>("pos");
        cv::Ptr<cv::ORB> orb = cv::ORB::create(nFeatures, scaleFactor, 8, edgeThreshold, 0, 2,
                                               cv::ORB::HARRIS_SCORE, 31, patchSize);
        cv::Mat imagecvin(h, w, CV_8UC3);
        cv::Mat imagecvgray(h, w, CV_8U);
        cv::Mat imagecvdetect(h, w, CV_8U);
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
        orb->detect(imagecvgray, ikeypoints);
//        zeno::log_info("orb->detect (imagecvin keypoints:{})", ikeypoints.size());
        orb->compute(imagecvgray, ikeypoints, imagecvdetect);
//        zeno::log_info("orb->compute(imagecvin, keypoints, imagecvout)");
//        zeno::log_info("ikeypoints.size{}",ikeypoints.size());
        if (ikeypoints.size() == 0) {
            throw zeno::Exception("Did not find any features");
        }
        cv::drawKeypoints(imagecvin, ikeypoints, imagecvout, cv::Scalar(255, 0, 0),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        // | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG
        zeno::log_info(
                "cv::drawKeypoints(imagecvin, keypoints, imagecvout, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);");

        cv::Mat imageFloat;
        imagecvdetect.convertTo(imageFloat, CV_32F, 1.0 / 255.0);
        for (const cv::KeyPoint &keypoint: ikeypoints) {
            float x = static_cast<float>(keypoint.pt.x);
            float y = static_cast<float>(keypoint.pt.y);
//            zeno::log_info("x{},y{}",x,y);
            image->uvs.push_back({x, y});
        }
        if (visualize) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    cv::Vec3b pixel = imagecvout.at<cv::Vec3b>(i, j);
                    image->verts[i * w + j][0] = zeno::min(static_cast<float>(pixel[0])/255,1.0f);
                    image->verts[i * w + j][1] = zeno::min(static_cast<float>(pixel[1])/255,1.0f);
                    image->verts[i * w + j][2] = zeno::min(static_cast<float>(pixel[2])/255,1.0f);
                    dp[i * w + j] = imageFloat.at<float>(i, j);
                }
            }
        }
        else{
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    dp[i * w + j] = imageFloat.at<float>(i, j);
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
        auto &dp = image->verts.add_attr<float>("descriptors");
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create(nFeatures, nOctaveLayers,
                                 contrastThreshold, edgeThreshold,sigma);
        cv::Mat imagecvin(h, w, CV_8UC3);
        cv::Mat imagecvgray(h, w, CV_8U);
        cv::Mat imagecvdetect(h, w, CV_8U);
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
        sift->detect(imagecvgray, ikeypoints);
        sift->compute(imagecvgray, ikeypoints, imagecvdetect);
//        zeno::log_info("ikeypoints.size{}",ikeypoints.size());
        if (ikeypoints.size() == 0) {
            throw zeno::Exception("Did not find any features");
        }
        cv::drawKeypoints(imagecvin, ikeypoints, imagecvout, cv::Scalar(255, 0, 0),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//                          | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG
//        zeno::log_info(
//                "cv::drawKeypoints(imagecvin, keypoints, imagecvout, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS "
//                "| cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);");

        cv::Mat imageFloat;
        imagecvdetect.convertTo(imageFloat, CV_32F, 1.0 / 255.0);
        for (const cv::KeyPoint &keypoint: ikeypoints) {
            float x = static_cast<float>(keypoint.pt.x);
            float y = static_cast<float>(keypoint.pt.y);
//            zeno::log_info("x{},y{}",x,y);
            image->uvs.push_back({x, y});
        }
        cv::Size ts = imageFloat.size();
        int tsize = ts.width * ts.height;
//        zeno::log_info("imageFloat(imagecvdetect)size:{}",tsize);
        if (visualize) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    cv::Vec3b pixel = imagecvout.at<cv::Vec3b>(i, j);
                    image->verts[i * w + j][0] = zeno::min(static_cast<float>(pixel[0])/255,1.0f);
                    image->verts[i * w + j][1] = zeno::min(static_cast<float>(pixel[1])/255,1.0f);
                    image->verts[i * w + j][2] = zeno::min(static_cast<float>(pixel[2])/255,1.0f);

                    dp[i * w + j] = imageFloat.at<float>(i, j);
                }
            }
        }
        else{
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    dp[i * w + j] = imageFloat.at<float>(i, j);
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

        auto &ud1 = image1->userData();
        int w1 = ud1.get2<int>("w");
        int h1 = ud1.get2<int>("h");
        auto &ud2 = image1->userData();
        int w2 = ud2.get2<int>("w");
        int h2 = ud2.get2<int>("h");
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(
                cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
        std::vector<cv::DMatch> matches;

        cv::Mat imagecvdescriptors1(h1, w1, CV_8U);
        cv::Mat imagecvdescriptors2(h2, w2, CV_8U);
        cv::Mat imagecvkeypoints1(h1, w1, CV_8U);
        cv::Mat imagecvkeypoints2(h2, w2, CV_8U);
        std::vector<cv::KeyPoint> keypoints;
        auto d1 = image1->verts.attr<float>("descriptors");
        auto d2 = image2->verts.attr<float>("descriptors");
        auto k1 = image1->uvs;
        auto k2 = image2->uvs;

        for (int i = 0; i < h1; i++) {
            for (int j = 0; j < w1; j++) {
                imagecvdescriptors1.at<uchar>(i, j) = d1[i * w1 + j];
            }
        }
        for (int i = 0; i < h2; i++) {
            for (int j = 0; j < w2; j++) {
                imagecvdescriptors2.at<uchar>(i, j) = d2[i * w2 + j];
            }
        }
        matcher->match(imagecvdescriptors1, imagecvdescriptors2, matches);
        std::vector<cv::Point2f> points1, points2;
        for (const auto &match: matches) {
            points1.push_back({k1[match.queryIdx][0], k1[match.queryIdx][1]});
            points2.push_back({k2[match.trainIdx][0], k2[match.trainIdx][1]});
        }
        //iphone11
        double focalLength = 26.0; // 焦距（单位：毫米）
        double sensorWidth = 5.715; // 传感器宽度（单位：毫米）
        double sensorHeight = 4.286; // 传感器高度（单位：毫米）
        double imageWidth = 4032.0; // 图像宽度（像素）
        double imageHeight = 3024.0; // 图像高度（像素）
//        double imageWidth = w;
//        double imageHeight = h;
        double fx = focalLength * (imageWidth / sensorWidth); // fx
        double fy = focalLength * (imageHeight / sensorHeight); // fy
        double cx = imageWidth / 2.0; // cx
        double cy = imageHeight / 2.0; // cy
        zeno::log_info("fx{},fy{},cx{},cy{}", fx, fy, cx, cy);
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                                                          0, fy, cy,
                                                          0, 0, 1);
        cv::Mat mask;
        cv::Mat essentialMatrix = cv::findEssentialMat(points1, points2, cameraMatrix, cv::RANSAC, 0.99, 1.0, mask);
        cv::Mat rotation, translation;
        cv::recoverPose(essentialMatrix, points1, points2, cameraMatrix, rotation, translation);
        auto &rt = image2->add_attr<float>("rotation");
        auto &tl = image2->add_attr<float>("translation");


        cv::Size ts = translation.size();
        for (int i = 0; i < ts.width * ts.height; i++) {
            tl[i] = translation.at<float>(i);
        }
        cv::Size rs = rotation.size();
        for (int i = 0; i < rs.width * rs.height; i++) {
            rt[i] = rotation.at<float>(i);
        }


        set_output("image", image2);
    }
};

ZENDEFNODE(ImageFeatureMatch, {
    {
        { "image1" },
        { "image2" },
    },
    {
        { "image" },
    },
    {},
    { "image" },
});

struct ImageStitching2 : INode {
    void apply() override {
        auto image1 = get_input<PrimitiveObject>("image1");
        auto image2 = get_input<PrimitiveObject>("image2");
        auto &ud1 = image1->userData();
        int w1 = ud1.get2<int>("w");
        int h1 = ud1.get2<int>("h");
        auto &ud2 = image2->userData();
        int w2 = ud2.get2<int>("w");
        int h2 = ud2.get2<int>("h");
        auto nFeatures = get_input2<float>("nFeatures");
        auto scaleFactor = get_input2<float>("scaleFactor");
        auto edgeThreshold = get_input2<float>("edgeThreshold");
        auto patchSize = get_input2<float>("patchSize");
//        cv::Ptr<cv::ORB> orb = cv::ORB::create(nFeatures, scaleFactor, 8, edgeThreshold, 0, 2, cv::ORB::HARRIS_SCORE, 31, patchSize);
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        cv::Mat imagecvin1(h1, w1, CV_8UC3);
        cv::Mat imagecvin2(h2, w2, CV_8UC3);
        cv::Mat imagecvgray1(h1, w1, CV_8U);
        cv::Mat imagecvgray2(h2, w2, CV_8U);
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
        cv::cvtColor(imagecvin1, imagecvgray1, cv::COLOR_RGB2GRAY);
        cv::cvtColor(imagecvin2, imagecvgray2, cv::COLOR_RGB2GRAY);
        std::vector<cv::KeyPoint> keypoints1, keypoints2;

        orb->detectAndCompute(imagecvgray1, cv::noArray(), keypoints1, descriptors1);
        orb->detectAndCompute(imagecvgray2, cv::noArray(), keypoints2, descriptors2);
        if (keypoints1.size() == 0 || keypoints2.size() == 0) {
            throw zeno::Exception("Did not find any features");
        }
        // 创建特征匹配器
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        // 对特征描述子进行匹配
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);
        // 筛选匹配点
        double minDist = 100.0;
        std::vector<cv::DMatch> goodMatches;
        for (const auto &match: matches) {
            if (match.distance < minDist) {
                goodMatches.push_back(match);
            }
        }
        // 提取匹配点的坐标
        std::vector<cv::Point2f> points1, points2;
        for (const auto &match: goodMatches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }
        cv::Mat fundamental = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3.0, 0.99);
        // 估计单应性矩阵
        cv::Mat homography = cv::findHomography(points1, points2, cv::RANSAC, 3.0);
        // 创建拼接后的图像
        cv::Mat result;
        cv::warpPerspective(imagecvin1, result, homography,
                            cv::Size(imagecvin1.cols + imagecvin2.cols, imagecvin1.rows));
        cv::Mat roi(result, cv::Rect(0, 0, imagecvin2.cols, imagecvin2.rows));
        imagecvin2.copyTo(roi);
//        cv::Mat result;
//        cv::stitching::Stitcher stitcher = cv::stitching::Stitcher::createDefault();

//        cv::Stitcher::Status status = stitcher.stitch(std::vector<cv::Mat>{image1, image2}, result);
        int w = result.cols;
        int h = result.rows;
        image1->verts.resize(w * h);
        ud1.set2("w", w);
        ud1.set2("h", h);

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                image1->verts[i * w + j][0] = result.at<cv::Vec3b>(i, j)[0] / 255;
                image1->verts[i * w + j][1] = result.at<cv::Vec3b>(i, j)[1] / 255;
                image1->verts[i * w + j][2] = result.at<cv::Vec3b>(i, j)[2] / 255;
            }
        }
        set_output("image", image1);
    }
};

ZENDEFNODE(ImageStitching2, {
    {
        { "image1" },
        { "image2" },
        { "float", "nFeatures", "100" },
        { "float", "scaleFactor", "1.2" },
        { "float", "edgeThreshold", "20" },
        { "float", "patchSize", "10" },
    },
    {
        { "image" },
    },
    {},
    { "deprecated" },
});

struct ImageStitching : INode {
    void apply() override {
//        auto primList = get_input<ListObject>("listPrim")->getRaw<PrimitiveObject>();
        auto image1 = get_input<PrimitiveObject>("image1");
        auto image2 = get_input<PrimitiveObject>("image2");
        auto mode = get_input2<std::string>("mode");
        auto &ud1 = image1->userData();
        int w1 = ud1.get2<int>("w");
        int h1 = ud1.get2<int>("h");
        auto &ud2 = image2->userData();
        int w2 = ud2.get2<int>("w");
        int h2 = ud2.get2<int>("h");
        auto nFeatures = get_input2<float>("nFeatures");
        auto scaleFactor = get_input2<float>("scaleFactor");
        auto edgeThreshold = get_input2<float>("edgeThreshold");
        auto patchSize = get_input2<float>("patchSize");
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