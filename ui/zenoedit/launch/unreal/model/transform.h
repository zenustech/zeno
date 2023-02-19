#ifndef ZENO_TRANSFORM_H
#define ZENO_TRANSFORM_H

#include <type_traits>

struct TestModel {
    int8_t a;

    template <class T>
    void pack(T &pack) {
        pack(a);
    }
};

#endif //ZENO_TRANSFORM_H
