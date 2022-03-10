#pragma once

#include <type_traits>
#include <array>
#ifdef __SSE__
#include <xmmintrin.h>
#endif

namespace zeno {
inline namespace streamed_write_h {

namespace details {
#ifdef __SSE__
template <class T, std::enable_if_t<sizeof(T) == 4, int> = 0>
static void stream_write(T *dst, T const *src) {
    _mm_stream_si32((int *)dst, *(int const *)src);
}

template <class T, std::enable_if_t<sizeof(T) == 16, int> = 0>
static void stream_write(T *dst, T const *src) {
    _mm_stream_ps((float *)dst, _mm_load_ps((float const *)src));
}

template <class T, std::enable_if_t<sizeof(T) != 4 && sizeof(T) % 4 == 0 && sizeof(T) % 16 != 0, int> = 0>
static void stream_write(T *dst, T const *src) {
    for (std::size_t i = 0; i < sizeof(T) / 4; i++)
        stream_write((float *)dst, (float const *)src + i);
}

template <class T, std::enable_if_t<sizeof(T) != 16 && sizeof(T) % 16 == 0, int> = 0>
static void stream_write(T *dst, T const *src) {
    for (std::size_t i = 0; i < sizeof(T) / 16; i++)
        stream_write((std::array<float, 4> *)dst + i*4, (std::array<float, 4> const *)src + i*4);
}
#else
template <class T>
static void stream_write(T *dst, T const *src) {
    std::memcpy((void *)dst, (void *)src, sizeof(T));
}
#endif
}

template <class T>
struct streamed {
    T &m_dst;

    streamed(T &dst) : m_dst(dst) {}

    streamed &operator=(T const &val) const {
        details::stream_write(m_dst, &val);
        return *this;
    }

    operator float() const {
        return m_dst;
    }
};

template <class T>
streamed(T &) -> streamed<T>;

template <class T>
struct streamed_array {
    T *m_data;

    streamed_array(T *data) : m_data(data) {}

    streamed<T> operator[](std::size_t i) const {
        return {m_data[i]};
    }
};

template <class T>
streamed_array(T *) -> streamed_array<T>;

}
}
