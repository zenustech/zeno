#pragma once

#include <cstring>
#include "zeno/zeno.h"

namespace zeno {

struct SimpleCharBuffer {
    ZENO_API SimpleCharBuffer(const char* InChar);

    ZENO_API SimpleCharBuffer(const char* InChar, size_t Size);

    ZENO_API SimpleCharBuffer(SimpleCharBuffer&& InBuffer) noexcept;

    ZENO_API SimpleCharBuffer& operator=(SimpleCharBuffer&& InBuffer) noexcept;

    ZENO_API ~SimpleCharBuffer();

    char* data;
    size_t length;

    template <class T>
    void pack(T& pack) {
        pack(data, length);
    }
};

extern "C" {
    struct SimpleCharBuffer;
}

}
