#ifndef ZENO_IMGCV_H
#define ZENO_IMGCV_H
#include <opencv2/core/utility.hpp>
#include "zeno/core/IObject.h"

namespace zeno {
    struct CVImageObject : IObjectClone<CVImageObject> {
        cv::Mat image;

        CVImageObject() = default;
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
        std::variant<cv::Mat> m;
    };
}
#endif //ZENO_IMGCV_H
