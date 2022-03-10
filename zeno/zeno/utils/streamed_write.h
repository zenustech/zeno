#pragma once

#include <type_traits>
#include <array>

namespace zeno {
inline namespace streamed_write_h {

namespace details {
#ifdef __SSE__
template <class T, class = std::enable_if_t<sizeof(T) == 4>>
static void stream_write(T *dst, T const *src) {
    _mm_stream_si32((int *)dst, *(int const *)src);
}

template <class T, class = std::enable_if_t<sizeof(T) == 16>>
static void stream_write(T *dst, T const *src) {
    _mm_stream_ps(dst, _mm_load_ps(src));
}

template <class T, class = std::enable_if_t<sizeof(T) % 4 == 0 && sizeof(T) % 16 != 0>>
static void stream_write(T *dst, T const *src) {
    for (std::size_t i = 0; i < sizeof(T) / 4; i++)
        _mm_stream_si32(i[(int *)dst], i[(int const *)src]);
}

template <class T, class = std::enable_if_t<sizeof(T) % 16 == 0>>
static void stream_write(T *dst, T const *src) {
    for (std::size_t i = 0; i < sizeof(T) / 16; i++)
        _mm_stream_ps((i * 4)[(float *)dst], (i * 4)[(float const *)src]);
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

    streamed operator[](std::size_t i) const {
        return {m_data[i]};
    }
};

template <class T>
streamed_array(T *) -> streamed_array<T>;

}
}
