#ifndef ZENO_IMAGE_PROC_H
#define ZENO_IMAGE_PROC_H
#include <zeno/utils/api.h>
#include <vector>
#include <zeno/utils/vec.h>

namespace zeno {
    ZENO_API std::vector<vec3f> float_gaussian_blur(const vec3f *data, int w, int h);
}

#endif //ZENO_IMAGE_PROC_H