#ifndef ZENO_TRANSFORM_H
#define ZENO_TRANSFORM_H

#include <type_traits>

struct Translation {
    float_t x, y, z;

    template <class T>
    void pack(T &pack) {
        pack(x, y, z);
    }
};

#endif //ZENO_TRANSFORM_H
