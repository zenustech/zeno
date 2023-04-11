#pragma once

namespace zeno {

struct SimpleCharBuffer {
    SimpleCharBuffer(const char* InChar) {
        length = std::strlen(InChar);
        data = new char[length];
        strncpy(data, InChar, length);
    }

    SimpleCharBuffer(const char* InChar, size_t Size) {
        length = Size + 1;
        data = new char[length];
        strncpy(data, InChar, length - 1);
        data[Size] = '\0';
    }

    SimpleCharBuffer(SimpleCharBuffer&& InBuffer) noexcept {
        length = InBuffer.length;
        data = InBuffer.data;
        InBuffer.data = nullptr;
        InBuffer.length = 0;
    }

    SimpleCharBuffer& operator=(SimpleCharBuffer&& InBuffer) noexcept {
        length = InBuffer.length;
        data = InBuffer.data;
        InBuffer.data = nullptr;
        InBuffer.length = 0;
        return *this;
    }

    ~SimpleCharBuffer() {
        delete []data;
    }

    char* data;
    size_t length;
};

extern "C" {
    struct SimpleCharBuffer;
}

}
