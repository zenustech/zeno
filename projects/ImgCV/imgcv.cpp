#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <zeno/zeno.h>

namespace zeno {

struct CVImageObject : IObjectClone<CVImageObject> {
    cv::Mat image;

    explicit CVImageObject(cv::Mat image) : image(std::move(image)) {}

    CVImageObject(CVImageObject &&) = default;
    CVImageObject &operator=(CVImageObject &&) = default;

    CVImageObject(CVImageObject const &img) : image(img.image.clone()) {
    }

    CVImageObject &operator=(CVImageObject const &img) {
        // notice that cv::Mat is shallow-copy, only .clone() will deep-copy
        image = img.image.clone();
        return *this;
    }

    cv::Mat
};

struct CVImageRead : INode {
    void apply() override {
        auto filename = get_input2<std::string>("filename");
        auto image = std::make_shared<CVImageObject>(cv::imread(filename, cv::IMREAD_COLOR));
        set_output("image", std::move(image));
    }
};

ZENDEFNODE(CVImageRead, {
    {
        {"string", "title", "CVImageShow"},
        {"CVImageObject", "image"},
    },
    {
        {"CVImageObject", "image"},
    },
    {},
    {"opencv"},
});

struct CVImageShow : INode {
    void apply() override {
        auto image = get_input<CVImageObject>("image");
        auto title = get_input2<std::string>("title");
        cv::imshow(title, image->image);
        set_output("image", std::move(image));
    }
};

ZENDEFNODE(CVImageShow, {
    {
        {"string", "title", "imshow"},
        {"CVImageObject", "image"},
    },
    {
        {"CVImageObject", "image"},
    },
    {},
    {"opencv"},
});

}
