#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <zeno/zeno.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/zeno_p.h>
#include "imgcv.h"
#include <zeno/utils/UserData.h>
namespace zeno {
namespace {

struct CVINode : INode {
    template <class To = double, class T>
    static auto tocvscalar(T const &val) {
        if constexpr (is_vec_n<T> == 4) {
            return cv::Scalar_<To>(val[3], val[2], val[1], val[0]);
        } else if constexpr (is_vec_n<T> == 3) {
            return cv::Scalar_<To>(val[2], val[1], val[0]);
        } else if constexpr (is_vec_n<T> == 2) {
            return cv::Scalar_<To>(val[1], val[0]);
        } else {
            return To(val);
        }
    }

    //template <class T>
    //static cv::_InputArray tocvinputarr(T const &val) {
        //if constexpr (is_vec_n<T> == 4) {
            //return cv::_InputArray(make_array(val[3], val[2], val[1], val[0]).data(), 4);
        //} else if constexpr (is_vec_n<T> == 3) {
            //return cv::_InputArray(make_array(val[2], val[1], val[0]).data(), 3);
        //} else if constexpr (is_vec_n<T> == 2) {
            //return cv::_InputArray(make_array(val[1], val[0]).data(), 2);
        //} else {
            //return cv::_InputArray((double)val);
        //}
    //}

    cv::Mat get_input_image(std::string const &name, bool inversed = false) {
        //if (has_input<NumericObject>(name)) {
            //auto num = get_input<NumericObject>(name);
            //bool is255 = has_input<NumericObject>("is255") && get_input2<bool>("is255");
            //return 155.f;
            //return std::visit([&] (auto const &val) -> cv::_InputArray {
                //auto tmp = inversed ? 1 - val : val;
                //return tocvinputarr(is255 ? tmp * 255 : tmp);
            //}, num->value);
        //} else {
            if (inversed) {
                cv::Mat newimg;
                auto img = get_input<CVImageObject>(name)->image;
                bool is255 = has_input<NumericObject>("is255") && get_input2<bool>("is255");
                if (is255) {
                    cv::bitwise_not(img, newimg);
                } else {
                    cv::invert(img, newimg);
                }
                return std::move(newimg);
            } else {
                return get_input<CVImageObject>(name)->image;
            }
        //}
    }
};

struct CVImageRead : CVINode {
    void apply() override {
        auto path = get_input2<std::string>("path");
        auto mode = get_input2<std::string>("mode");
        auto is255 = get_input2<bool>("is255");
        cv::ImreadModes flags = array_lookup(
            {cv::IMREAD_COLOR, cv::IMREAD_GRAYSCALE, cv::IMREAD_UNCHANGED, cv::IMREAD_UNCHANGED},
            array_index_safe({"RGB", "GRAY", "RGBA", "UNCHANGED"}, mode, "mode"));
        auto image = std::make_shared<CVImageObject>(cv::imread(path, flags));
        if (image->image.empty()) {
            zeno::log_error("opencv failed to read image file: {}", path);
        }
        if (mode == "RGBA") {
            if (is255) {
                image->image.convertTo(image->image, 
                                       image->image.channels() == 1 ? CV_8UC1 :
                                       image->image.channels() == 2 ? CV_8UC2 :
                                       image->image.channels() == 3 ? CV_8UC3 :
                                       CV_8UC4);
            } else {
                image->image.convertTo(image->image, 
                                       image->image.channels() == 1 ? CV_32FC1 :
                                       image->image.channels() == 2 ? CV_32FC2 :
                                       image->image.channels() == 3 ? CV_32FC3 :
                                       CV_32FC4);
            }
            if (image->image.channels() == 1) {
                cv::cvtColor(image->image, image->image, cv::COLOR_GRAY2BGRA);
            } else if (image->image.channels() == 3) {
                cv::cvtColor(image->image, image->image, cv::COLOR_BGR2BGRA);
            }
        } else {
            if (!is255) {
                image->image.convertTo(image->image, 
                                       image->image.channels() == 1 ? CV_32FC1 :
                                       image->image.channels() == 2 ? CV_32FC2 :
                                       image->image.channels() == 3 ? CV_32FC3 :
                                       CV_32FC4);
            }
        }
        set_output("image", std::move(image));
    }
};

ZENDEFNODE(CVImageRead, {
    {
        {"readpath", "path", ""},
        {"enum RGB GRAY RGBA UNCHANGED", "mode", "RGB"},
        {"bool", "is255", "1"},
    },
    {
        {"CVImageObject", "image"},
    },
    {},
    {"opencv"},
});

struct CVSepAlpha : CVINode {
    void apply() override {
        auto image = get_input<CVImageObject>("imageRGBA");
        auto imageRGB = std::make_shared<CVImageObject>();
        cv::cvtColor(image->image, imageRGB->image, cv::COLOR_BGRA2BGR);
        std::vector<cv::Mat> channels;
        cv::split(image->image, channels);
        auto imageAlpha = std::make_shared<CVImageObject>(channels.back());
        if (!get_input2<bool>("alphaAsGray")) {
            cv::cvtColor(imageAlpha->image, imageAlpha->image, cv::COLOR_GRAY2BGR);
        }
        set_output("imageRGB", std::move(imageRGB));
        set_output("imageAlpha", std::move(imageAlpha));
    }
};

ZENDEFNODE(CVSepAlpha, {
    {
        {"CVImageObject", "imageRGBA"},
        {"bool", "alphaAsGray", "0"},
    },
    {
        {"CVImageObject", "imageRGB"},
        {"CVImageObject", "imageAlpha"},
    },
    {},
    {"opencv"},
});

struct CVImageSepRGB : CVINode {
    void apply() override {
        auto image = get_input<CVImageObject>("imageRGB");
        std::vector<cv::Mat> channels;
        cv::split(image->image, channels);
        auto imageB = std::make_shared<CVImageObject>(channels.at(0));
        auto imageG = std::make_shared<CVImageObject>(channels.at(1));
        auto imageR = std::make_shared<CVImageObject>(channels.at(2));
        set_output("imageR", std::move(imageR));
        set_output("imageG", std::move(imageG));
        set_output("imageB", std::move(imageG));
    }
};

ZENDEFNODE(CVImageSepRGB, {
    {
        {"CVImageObject", "imageRGB"},
    },
    {
        {"CVImageObject", "imageR"},
        {"CVImageObject", "imageG"},
        {"CVImageObject", "imageB"},
    },
    {},
    {"opencv"},
});

struct CVImageShow : CVINode {
    void apply() override {
        auto image = get_input_image("image");
        auto title = get_input2<std::string>("title");
        cv::imshow(title, image);
        if (get_input2<bool>("waitKey"))
            cv::waitKey();
    }
};

ZENDEFNODE(CVImageShow, {
    {
        {"CVImageObject", "image"},
        {"string", "title", "imshow"},
        {"bool", "waitKey", "1"},
    },
    {
    },
    {},
    {"opencv"},
});

struct CVWaitKey : CVINode {
    void apply() override {
        auto delay = get_input2<int>("delay");
        int kc = cv::waitKey(delay);
        set_output2("hasPressed", kc != -1);
        set_output2("keyCode", kc);
    }
};

ZENDEFNODE(CVWaitKey, {
    {
        {"int", "delay", "0"},
    },
    {
        {"bool", "hasPressed"},
        {"int", "keyCode"},
    },
    {},
    {"opencv"},
});

struct CVImageAdd : CVINode {
    void apply() override {
        auto image1 = get_input_image("image1");
        auto image2 = get_input_image("image2");
        auto weight1 = get_input2<float>("weight1");
        auto weight2 = get_input2<float>("weight2");
        auto constant = get_input2<float>("constant");
        auto resimage = std::make_shared<CVImageObject>();
        if (weight1 == 1 && weight2 == 1 && constant == 0) {
            cv::add(image1, image2, resimage->image);
        } else {
            cv::addWeighted(image1, weight1, image2, weight2, constant, resimage->image);
        }
        set_output("resimage", std::move(resimage));
    }
};

struct CVImageSubtract : CVINode {
    void apply() override {
        auto image1 = get_input_image("image1");
        auto image2 = get_input_image("image2");
        auto resimage = std::make_shared<CVImageObject>();
        cv::subtract(image1, image2, resimage->image);
        set_output("resimage", std::move(resimage));
    }
};

ZENDEFNODE(CVImageSubtract, {
    {
        {"CVImageObject", "image1"},
        {"CVImageObject", "image2"},
    },
    {
        {"CVImageObject", "resimage"},
    },
    {},
    {"opencv"},
});

struct CVImageMultiply : CVINode {
    void apply() override {
        auto image1 = get_input_image("image");
        auto inverse = get_input2<bool>("inverse");
        auto is255 = get_input2<bool>("is255");
        if (has_input<NumericObject>("factor")) {
            auto factor = get_input2<float>("factor");
            if (inverse) factor = 1 - factor;
            if (is255) factor = 255 * factor;
            auto resimage = std::make_shared<CVImageObject>();
            cv::multiply(image1, factor, resimage->image, is255 ? 1.f / 255.f : 1.f);
            set_output("resimage", std::move(resimage));
        } else {
            auto image2 = get_input_image("factor", inverse);
            auto resimage = std::make_shared<CVImageObject>();
            cv::multiply(image1, image2, resimage->image, is255 ? 1.f / 255.f : 1.f);
            set_output("resimage", std::move(resimage));
        }
    }
};

ZENDEFNODE(CVImageMultiply, {
    {
        {"CVImageObject", "image"},
        {"float"/*or CVImageObject*/, "factor", "1"},
        {"bool", "inverse", "0"},
        {"bool", "is255", "1"},
    },
    {
        {"CVImageObject", "resimage"},
    },
    {},
    {"opencv"},
});

struct CVImageDivide : CVINode {
    void apply() override {
        auto image1 = get_input_image("image");
        auto inverse = get_input2<bool>("inverse");
        auto image2 = get_input_image("factor", inverse);
        auto is255 = get_input2<bool>("is255");
        auto resimage = std::make_shared<CVImageObject>();
        cv::divide(image1, image2, resimage->image, is255 ? 1.f / 255.f : 1.f);
        set_output("resimage", std::move(resimage));
    }
};

ZENDEFNODE(CVImageDivide, {
    {
        {"CVImageObject", "image"},
        {"float"/*or CVImageObject*/, "factor", "1"},
        {"bool", "inverse", "0"},
        {"bool", "is255", "1"},
    },
    {
        {"CVImageObject", "resimage"},
    },
    {},
    {"opencv"},
});

struct CVImageBlend : CVINode {
    void apply() override {
        auto image1 = get_input_image("image1");
        auto image2 = get_input_image("image2");
        auto is255 = get_input2<bool>("is255");
        auto inverse = get_input2<bool>("inverse");
        auto resimage = std::make_shared<CVImageObject>();
        if (inverse) {
            std::swap(image1, image2);
        }
        if (has_input<NumericObject>("factor")) {
            auto factor = get_input2<float>("factor");
            cv::addWeighted(image1, 1 - factor, image2, factor, 0, resimage->image);
        } else {
            auto factor = get_input_image("factor");
            cv::Mat factorinv, tmp1, tmp2;
            if (is255) {
                cv::bitwise_not(factor, factorinv);
            } else {
                cv::invert(factor, factorinv);
            }
            cv::multiply(image1, factorinv, tmp1, is255 ? 1.f / 255.f : 1.f);
            cv::multiply(image2, factor, tmp2, is255 ? 1.f / 255.f : 1.f);
            cv::add(tmp1, tmp2, resimage->image);
        }
        set_output("resimage", std::move(resimage));
    }
};

ZENDEFNODE(CVImageBlend, {
    {
        {"CVImageObject", "image1"},
        {"CVImageObject", "image2"},
        {"float"/*or CVImageObject*/, "factor", "0.5"},
        {"bool", "inverse", "0"},
        {"bool", "is255", "1"},
    },
    {
        {"CVImageObject", "resimage"},
    },
    {},
    {"opencv"},
});

struct CVImageInvert : CVINode {
    void apply() override {
        auto image = get_input_image("image");
        auto is255 = get_input2<bool>("is255");
        auto resimage = std::make_shared<CVImageObject>();
        if (is255) {
            cv::bitwise_not(image, resimage->image);
        } else {
            cv::invert(image, resimage->image);
        }
        set_output("resimage", std::move(resimage));
    }
};

ZENDEFNODE(CVImageInvert, {
    {
        {"CVImageObject", "image"},
        {"bool", "is255", "1"},
    },
    {
        {"CVImageObject", "resimage"},
    },
    {},
    {"opencv"},
});

struct CVConvertColor : CVINode {
    void apply() override {
        auto image = get_input_image("image");
        auto mode = get_input2<std::string>("mode");
        cv::ColorConversionCodes code = array_lookup({
            cv::COLOR_BGR2GRAY,
            cv::COLOR_GRAY2BGR,
            cv::COLOR_BGR2RGB,
            cv::COLOR_BGR2BGRA,
            cv::COLOR_BGRA2BGR,
            cv::COLOR_BGR2HSV,
            cv::COLOR_HSV2BGR,
        }, array_index_safe({
            "BGR2GRAY",
            "GRAY2BGR",
            "BGR2RGB",
            "BGR2BGRA",
            "BGRA2BGR",
            "BGR2HSV",
            "HSV2BGR",
        }, mode, "mode"));
        auto resimage = std::make_shared<CVImageObject>();
        cv::cvtColor(image, resimage->image, code);
        set_output("resimage", std::move(resimage));
    }
};

ZENDEFNODE(CVConvertColor, {
    {
        {"CVImageObject", "image"},
        {
            "enum "
            "BGR2GRAY "
            "GRAY2BGR "
            "BGR2RGB "
            "BGR2BGRA "
            "BGRA2BGR "
            "BGR2HSV "
            "HSV2BGR "
            , "mode", "GRAY2BGR"},
    },
    {
        {"CVImageObject", "resimage"},
    },
    {},
    {"opencv"},
});

struct CVImageGrayscale : CVINode {
    void apply() override {
        auto image = get_input_image("image");
        auto resimage = std::make_shared<CVImageObject>();
        cv::Mat tmp;
        cv::cvtColor(image, tmp, cv::COLOR_BGR2GRAY);
        cv::cvtColor(tmp, resimage->image, cv::COLOR_GRAY2BGR);
        set_output("resimage", std::move(resimage));
    }
};

ZENDEFNODE(CVImageGrayscale, {
    {
        {"CVImageObject", "image"},
    },
    {
        {"CVImageObject", "resimage"},
    },
    {},
    {"opencv"},
});

struct CVImageFillColor : CVINode {
    void apply() override {
        auto likeimage = get_input<CVImageObject>("image");
        auto is255 = get_input2<bool>("is255");
        auto color = tocvscalar<float>(get_input2<zeno::vec3f>("color"));
        auto image = get_input2<bool>("inplace") ? likeimage
            : std::make_shared<CVImageObject>(likeimage->image.clone());
        if (has_input("mask")) {
            auto mask = get_input<CVImageObject>("mask");
            if (is255) {
                cv::Point3_<unsigned char> cval;
                cval.x = (unsigned char)std::clamp(color[0] * 255.f, 0.f, 255.f);
                cval.y = (unsigned char)std::clamp(color[1] * 255.f, 0.f, 255.f);
                cval.z = (unsigned char)std::clamp(color[2] * 255.f, 0.f, 255.f);
                image->image.setTo(cv::Scalar(cval.x, cval.y, cval.z), mask->image);
            } else {
                image->image.setTo(cv::Scalar(color[0], color[1], color[2]), mask->image);
            }
        } else {
            if (is255) {
                cv::Point3_<unsigned char> cval;
                cval.x = (unsigned char)std::clamp(color[0] * 255.f, 0.f, 255.f);
                cval.y = (unsigned char)std::clamp(color[1] * 255.f, 0.f, 255.f);
                cval.z = (unsigned char)std::clamp(color[2] * 255.f, 0.f, 255.f);
                image->image.setTo(cv::Scalar(cval.x, cval.y, cval.z));
            } else {
                image->image.setTo(cv::Scalar(color[0], color[1], color[2]));
            }
        }
        set_output("image", std::move(image));
    }
};

ZENDEFNODE(CVImageFillColor, {
    {
        {"CVImageObject", "image"},
        {"optional CVImageObject", "mask"},
        {"bool", "is255", "1"},
        {"vec3f", "color", "1,1,1"},
        {"bool", "inplace", "0"},
    },
    {
        {"CVImageObject", "image"},
    },
    {},
    {"opencv"},
});

struct CVImageMaskedAssign : CVINode {
    void apply() override {
        auto likeimage = get_input<CVImageObject>("image");
        auto srcimage = get_input<CVImageObject>("srcImage");
        auto is255 = get_input2<bool>("is255");
        auto image = get_input2<bool>("inplace") ? likeimage
            : std::make_shared<CVImageObject>(likeimage->image.clone());
        if (has_input("mask")) {
            auto mask = get_input<CVImageObject>("mask");
            image->image.setTo(srcimage->image, mask->image);
        } else {
            image->image.setTo(srcimage->image);
        }
        set_output("image", std::move(image));
    }
};

ZENDEFNODE(CVImageMaskedAssign, {
    {
        {"CVImageObject", "image"},
        {"CVImageObject", "srcImage"},
        {"optional CVImageObject", "mask"},
        {"bool", "is255", "1"},
        {"bool", "inplace", "0"},
    },
    {
        {"CVImageObject", "image"},
    },
    {},
    {"opencv"},
});

struct CVImageBlit : CVINode {
    void apply() override {
        auto likeimage = get_input<CVImageObject>("image");
        auto srcimage = get_input<CVImageObject>("srcImage");
        auto is255 = get_input2<bool>("is255");
        auto centered = get_input2<bool>("centered");
        auto image = get_input2<bool>("inplace") ? likeimage
            : std::make_shared<CVImageObject>(likeimage->image.clone());
        auto x0 = get_input2<int>("X0");
        auto y0 = get_input2<int>("Y0");
        auto dx = srcimage->image.cols;
        auto dy = srcimage->image.rows;
        auto maxx = image->image.cols;
        auto maxy = image->image.rows;
        if (centered) {
            x0 += dx / 2;
            y0 += dy / 2;
        }
        //zeno::log_warn("dx {} dy {}", dx, dy);
        int sx0 = 0, sy0 = 0;
        bool hasmodroi = false;
        if (x0 < 0) {
            dx -= -x0;
            sx0 = -x0;
            x0 = 0;
            hasmodroi = true;
        }
        if (y0 < 0) {
            dy -= -y0;
            sy0 = -y0;
            y0 = 0;
            hasmodroi = true;
        }
        if (x0 + dx > maxx) {
            dx = maxx - x0;
            hasmodroi = true;
        }
        if (y0 + dy > maxy) {
            dy = maxy - y0;
            hasmodroi = true;
        }
        //zeno::log_warn("x0 {} y0 {} dx {} dy {} sx0 {} sy0 {}", x0, y0, dx, dy, sx0, sy0);
        cv::Rect roirect(x0, y0, dx, dy);
        auto roi = image->image(roirect);
        auto srcroi = srcimage->image;
        if (hasmodroi) {
            srcroi = srcroi(cv::Rect(sx0, sy0, dx, dy));
        }
        if (has_input("mask")) {
            auto mask = get_input<CVImageObject>("mask");
            auto factor = mask->image;
            if (hasmodroi) {
                factor = factor(cv::Rect(sx0, sy0, dx, dy));
            }
            if (get_input2<bool>("isAlphaMask")) {
                auto image1 = roi, image2 = srcroi;
                cv::Mat factorinv, tmp1, tmp2;
                if (is255) {
                    cv::bitwise_not(factor, factorinv);
                } else {
                    cv::invert(factor, factorinv);
                }
                cv::multiply(image1, factorinv, tmp1, is255 ? 1.f / 255.f : 1.f);
                cv::multiply(image2, factor, tmp2, is255 ? 1.f / 255.f : 1.f);
                cv::add(tmp1, tmp2, tmp2);
                tmp2.copyTo(roi);
            } else {
                srcroi.copyTo(roi, factor);
            }
        } else {
            srcroi.copyTo(roi);
        }
        set_output("image", std::move(image));
    }
};

ZENDEFNODE(CVImageBlit, {
    {
        {"CVImageObject", "image"},
        {"CVImageObject", "srcImage"},
        {"int", "X0", "0"},
        {"int", "Y0", "0"},
        {"bool", "centered", "0"},
        {"optional CVImageObject", "mask"},
        {"bool", "isAlphaMask", "1"},
        {"bool", "is255", "1"},
        {"bool", "inplace", "0"},
    },
    {
        {"CVImageObject", "image"},
    },
    {},
    {"opencv"},
});

struct CVImageCrop : CVINode {
    void apply() override {
        auto srcimage = get_input<CVImageObject>("srcimage");
        auto is255 = get_input2<bool>("is255");
        auto isDeep = get_input2<bool>("deepCopy");
        auto x0 = get_input2<int>("X0");
        auto y0 = get_input2<int>("Y0");
        auto dx = get_input2<int>("DX");
        auto dy = get_input2<int>("DY");
        cv::Rect roirect(x0, y0, dx, dy);
        auto roi = srcimage->image(roirect);
        if (isDeep) roi = roi.clone();
        auto image = std::make_shared<CVImageObject>(std::move(roi));
        set_output("image", std::move(image));
    }
};

ZENDEFNODE(CVImageCrop, {
    {
        {"CVImageObject", "srcimage"},
        {"int", "X0", "0"},
        {"int", "Y0", "0"},
        {"int", "DX", "32"},
        {"int", "DY", "32"},
        {"bool", "is255", "1"},
        {"bool", "deepCopy", "1"},
    },
    {
        {"CVImageObject", "image"},
    },
    {},
    {"opencv"},
});

struct CVMakeImage : CVINode {
    void apply() override {
        auto likeimage = get_input<CVImageObject>("image");
        auto srcimage = get_input<CVImageObject>("srcImage");
        auto mode = get_input2<std::string>("mode");
        auto isWhite = get_input2<bool>("whiteBg");
        auto is255 = get_input2<bool>("is255");
        auto w = get_input2<int>("width");
        auto h = get_input2<int>("height");
        int ty = array_lookup(is255 ?
                              make_array(CV_8UC3, CV_8UC1, CV_8UC4) :
                              make_array(CV_32FC3, CV_32FC1, CV_32FC4),
            array_index_safe({"RGB", "GRAY", "RGBA"}, mode, "mode"));
        auto image = std::make_shared<CVImageObject>(cv::Mat(h, w, ty, cv::Scalar::all(
                    isWhite ? (is255 ? 1 : 255) : 0)));
        set_output("image", std::move(image));
    }
};

ZENDEFNODE(CVMakeImage, {
    {
        {"int", "width", "512"},
        {"int", "height", "512"},
        {"enum RGB GRAY RGBA", "mode", "RGB"},
        {"bool", "whiteBg", "0"},
        {"bool", "is255", "1"},
    },
    {
        {"CVImageObject", "image"},
    },
    {},
    {"opencv"},
});

struct CVGetImageSize : CVINode {
    void apply() override {
        auto image = get_input<CVImageObject>("image");
        set_output2("width", image->image.cols);
        set_output2("height", image->image.rows);
        set_output2("channels", image->image.channels());
    }
};

ZENDEFNODE(CVGetImageSize, {
    {
        {"CVImageObject", "image"},
    },
    {
        {"int", "width"},
        {"int", "height"},
        {"int", "channels"},
    },
    {},
    {"opencv"},
});

struct CVImageFillGrad : CVINode {
    void apply() override {
        auto likeimage = get_input<CVImageObject>("image");
        auto angle = get_input2<float>("angle");
        auto scale = get_input2<float>("scale");
        auto offset = get_input2<float>("offset");
        auto is255 = get_input2<bool>("is255");
        auto color1 = tocvscalar<float>(get_input2<zeno::vec3f>("color1"));
        auto color2 = tocvscalar<float>(get_input2<zeno::vec3f>("color2"));
        auto image = get_input2<bool>("inplace") ? likeimage
            : std::make_shared<CVImageObject>(likeimage->image.clone());
        vec2i shape(image->image.size[1], image->image.size[0]);
        vec2f invshape = 1.f / shape;
        angle *= (std::atan(1.f) * 4) / 180;
        vec2f dir(std::cos(angle), std::sin(angle));
        auto invscale = 0.5f / scale;
        auto neoffset = 0.5f - (offset * 2 - 1) * invscale;
        if (is255) {
            image->image.forEach<cv::Point3_<unsigned char>>([&] (cv::Point3_<unsigned char> &val, const int *pos) {
                vec2i posv(pos[1], pos[0]);
                float f = dot(posv * invshape * 2 - 1, dir) * invscale + neoffset, omf = 1 - f;
                val.x = (unsigned char)std::clamp((omf * color1[0] + f * color2[0]) * 255.f, 0.f, 255.f);
                val.y = (unsigned char)std::clamp((omf * color1[1] + f * color2[1]) * 255.f, 0.f, 255.f);
                val.z = (unsigned char)std::clamp((omf * color1[2] + f * color2[2]) * 255.f, 0.f, 255.f);
            });
        } else {
            image->image.forEach<cv::Point3_<float>>([&] (cv::Point3_<float> &val, const int *pos) {
                vec2i posv(pos[1], pos[0]);
                float f = dot(posv * invshape * 2 - 1, dir) * invscale + neoffset, omf = 1 - f;
                val.x = omf * color1[0] + f * color2[0];
                val.y = omf * color1[1] + f * color2[1];
                val.z = omf * color1[2] + f * color2[2];
            });
        }
        set_output("image", std::move(image));
    }
};

ZENDEFNODE(CVImageFillGrad, {
    {
        {"CVImageObject", "image"},
        {"float", "angle", "0"},     // rotation clock-wise
        {"float", "scale", "1"},     // thickness of gradient
        {"float", "offset", "0.5"},  // 0 to 1
        {"bool", "is255", "1"},
        {"vec3f", "color1", "0,0,0"},
        {"vec3f", "color2", "1,1,1"},
        {"bool", "inplace", "0"},
    },
    {
        {"CVImageObject", "image"},
    },
    {},
    {"opencv"},
});

struct CVImageDrawPoly : CVINode {
    void apply() override {
        auto image = get_input<CVImageObject>("image");
        auto color = tocvscalar<float>(get_input2<zeno::vec3f>("color"));
        if (!get_input2<bool>("inplace"))
            image = std::make_shared<CVImageObject>(*image);
        auto prim = get_input<PrimitiveObject>("prim");
        auto linewidth = get_input2<int>("linewidth");
        auto batched = get_input2<bool>("batched");
        auto antialias = get_input2<bool>("antialias");
        auto is255 = get_input2<bool>("is255");
        if (is255) color *= 255.f;
            //image->image.setTo(cv::Scalar::all(0));
        vec2i shape(image->image.size[1], image->image.size[0]);

        std::vector<std::vector<cv::Point>> vpts(prim->polys.size());
        for (int i = 0; i < prim->polys.size(); i++) {
            auto [base, len] = prim->polys[i];
            auto &pt = vpts[i];
            pt.resize(len);
            for (int k = 0; k < len; k++) {
                auto v = prim->verts[prim->loops[base + k]];
                pt[k].x = int((v[0] * 0.5f + 0.5f) * shape[0]);
                pt[k].y = int((v[1] * -0.5f + 0.5f) * shape[1]);
            }
        }
        std::vector<const cv::Point *> pts(vpts.size());
        std::vector<int> npts(vpts.size());
        for (int i = 0; i < vpts.size(); i++) {
            pts[i] = vpts[i].data();
            npts[i] = vpts[i].size();
        }

        cv::LineTypes linemode = antialias ? cv::LINE_AA : cv::LINE_4;
        if (linewidth > 0) {
            if (batched) {
                cv::polylines(image->image, pts.data(), npts.data(), npts.size(), 0, color, linewidth, linemode);
            } else {
                for (int i = 0; i < npts.size(); i++) {
                    cv::polylines(image->image, pts.data() + i, npts.data() + i, 1, 0, color, linewidth, linemode);
                }
            }
        } else {
            if (batched) {
                cv::fillPoly(image->image, pts.data(), npts.data(), npts.size(), color, linemode);
            } else {
                for (int i = 0; i < npts.size(); i++) {
                    cv::fillPoly(image->image, pts.data() + i, npts.data() + i, 1, color, linemode);
                }
            }
        }
        set_output("image", std::move(image));
    }
};

ZENDEFNODE(CVImageDrawPoly, {
    {
        {"CVImageObject", "image"},
        {"PrimitiveObject", "prim"},
        {"vec3f", "color", "1,1,1"},
        {"PrimitiveObject", "points"},
        {"int", "linewidth", "0"},
        {"bool", "inplace", "0"},
        {"bool", "batched", "0"},
        {"bool", "antialias", "0"},
        {"bool", "is255", "1"},
    },
    {
        {"CVImageObject", "image"},
    },
    {},
    {"opencv"},
});

struct CVImagePutText : CVINode {
    void apply() override {
        auto likeimage = get_input<CVImageObject>("image");
        auto image = get_input2<bool>("inplace") ? likeimage
            : std::make_shared<CVImageObject>(likeimage->image.clone());
        auto text = get_input2<std::string>("text");
        auto fontFace = get_input2<int>("fontFace");
        auto thickness = get_input2<int>("thickness");
        auto antialias = get_input2<bool>("antialias");
        auto scale = get_input2<float>("scale");
        auto is255 = get_input2<bool>("is255");
        auto color = tocvscalar<double>(get_input2<zeno::vec3f>("color") * (is255 ? 255 : 1));
        cv::Point org(get_input2<int>("X0"), get_input2<int>("Y0"));
        cv::putText(image->image, text, org, fontFace, scale, color,
                    thickness, antialias ? cv::LINE_AA : cv::LINE_8);
        set_output("resimage", std::move(image));
    }
};

ZENDEFNODE(CVImagePutText, {
    {
        {"CVImageObject", "image"},
        {"string", "text", "Hello, World"},
        {"int", "X0", "0"},
        {"int", "Y0", "0"},
        {"bool", "is255", "1"},
        {"vec3f", "color", "1,1,1"},
        {"float", "scale", "1"},
        {"int", "thickness", "1"},
        {"int", "fontFace", "0"},
        {"bool", "antialias", "0"},
        {"bool", "inplace", "0"},
    },
    {
        {"CVImageObject", "resimage"},
    },
    {},
    {"opencv"},
});

struct ReadImageByOpenCV : INode {
  void apply() override {
    auto inputPath = get_input2<std::string>("inputPath");
    cv::Mat image = cv::imread(inputPath, cv::IMREAD_UNCHANGED);
    cv::Mat exrImage;
    image.convertTo(exrImage, CV_32F);
    int width = image.cols;
    int height = image.rows;
    auto img = std::make_shared<PrimitiveObject>();
    img->verts.resize(width * height);
    auto &clr = img->verts.add_attr<vec3f>("clr");
    for (size_t j = 0; j < height; j++) {
      for (size_t i = 0; i < width; i++) {
        size_t index = j * width + i;
        img->verts[index] = {float(i) / float(width), float(j) / float(height), 0};
        float Y = exrImage.at<float>(j, i);
        clr[index] = {Y, Y, Y};
      }
    }
    img->userData().set2("w", width);
    img->userData().set2("h", height);

    set_output("image", std::move(img));
  }
};

ZENDEFNODE(ReadImageByOpenCV, {
                                  {
                                      {"readpath", "inputPath"},
                                  },
                                  {
                                      "image",
                                  },
                                  {},
                                  {"opencv"},
                              });
}

}
