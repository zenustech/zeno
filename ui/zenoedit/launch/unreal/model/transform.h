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

ADD_TYPE_ID_FOR_TYPE(TestModel);

#endif //ZENO_TRANSFORM_H
