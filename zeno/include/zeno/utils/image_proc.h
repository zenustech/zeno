#ifndef ZENO_IMAGE_PROC_H
#define ZENO_IMAGE_PROC_H
#include <zeno/utils/api.h>
#include <vector>
#include <zeno/utils/vec.h>

namespace zeno {
template<typename T>
void image_flip_vertical(T *v, int w, int h) {
    for (auto j = 0; j < h / 2; j++) {
        for (auto i = 0; i < w; i++) {
            auto index1 = i + j * w;
            auto index2 = i + (h - j - 1) * w;
            std::swap(v[index1], v[index2]);
        }
    }
}
template<typename T>
void image_flip_horizontal(T *v, int w, int h) {
    for (auto j = 0; j < h; j++) {
        for (auto i = 0; i < w / 2; i++) {
            auto index1 = i + j * w;
            auto index2 = j * w + (w - i - 1);
            std::swap(v[index1], v[index2]);
        }
    }
}
    ZENO_API std::vector<vec3f> float_gaussian_blur(const vec3f *data, int w, int h);
}

#endif //ZENO_IMAGE_PROC_H