#pragma once

#include <cstring>
#include "zeno/zeno.h"

namespace zeno {

struct SimpleCharBuffer {
    SimpleCharBuffer(const char* InChar);

    SimpleCharBuffer(const char* InChar, size_t Size);

    SimpleCharBuffer(SimpleCharBuffer&& InBuffer) noexcept;

    SimpleCharBuffer& operator=(SimpleCharBuffer&& InBuffer) noexcept;

    ZENO_API ~SimpleCharBuffer();

    char* data;
    size_t length;
};

extern "C" {
    struct SimpleCharBuffer;
}

}
