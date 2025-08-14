#include <opencv2/core/utility.hpp>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/scope_exit.h>
#include <stdexcept>
#include <zeno/utils/log.h>
#include <opencv2/videoio.hpp>
#include <filesystem>
#include <opencv2/opencv.hpp>
#if 0

using namespace cv;

namespace zeno {

namespace {

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
                image->verts[(h - i - 1) * w + j] = {rgb[2] / 255, rgb[1] / 255, rgb[0] / 255};
            }
        }
        set_output("image", image);
    }
};

ZENDEFNODE(ReadImageFromVideo, {
    {
        { "readpath", "path" },
        { "int", "frame", "1" },
    },
    {
        { "PrimitiveObject", "image" },
    },
    {},
    { "image" },
});

}
}

#endif