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

static void sobel(std::shared_ptr<PrimitiveObject> & grayImage, int width, int height, std::vector<float>& dx, std::vector<float>& dy)
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
    { "image" },
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
    { "deprecated" },
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