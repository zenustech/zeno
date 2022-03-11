#pragma once

#include <array>
#include <cstring>
#include <type_traits>
#if defined(__SSE2__) && __has_include(<emmintrin.h>)
#include <emmintrin.h>
#endif

namespace zeno {
inline namespace streamed_write_h {

namespace details {
#if defined(__SSE2__) && __has_include(<emmintrin.h>)
template <class T, std::enable_if_t<sizeof(T) == 4, int> = 0>
static void stream_write(T *dst, T const *src) {
    _mm_stream_si32((int *)dst, *(int const *)src);
}

template <class T, std::enable_if_t<sizeof(T) == 16 && alignof(T) == 16, int> = 0>
static void stream_write(T *dst, T const *src) {
    _mm_stream_ps((float *)dst, _mm_load_ps((float const *)src));
}

template <class T, std::enable_if_t<sizeof(T) == 16 && alignof(T) != 16, int> = 0>
static void stream_write(T *dst, T const *src) {
    _mm_stream_ps((float *)dst, _mm_loadu_ps((float const *)src));
}

template <class T, std::enable_if_t<sizeof(T) != 4 && sizeof(T) % 4 == 0 && sizeof(T) % 16 != 0, int> = 0>
static void stream_write(T *dst, T const *src) {
    for (std::size_t i = 0; i < sizeof(T) / 4; i++)
        stream_write((float *)dst + i, (float const *)src + i);
}

template <class T, std::enable_if_t<sizeof(T) != 16 && sizeof(T) % 16 == 0, int> = 0>
static void stream_write(T *dst, T const *src) {
    for (std::size_t i = 0; i < sizeof(T) / 16; i++)
        stream_write((std::array<float, 4> *)dst + i, (std::array<float, 4> const *)src + i);
}
#else
template <class T>
static void stream_write(T *dst, T const *src) {
    std::memcpy((void *)dst, (void const *)src, sizeof(T));
}
#endif
}

template <class T>
struct streamed {
    static_assert(std::is_same_v<T, std::decay_t<T>>);
    static_assert(std::is_pod_v<T>);

    T &m_dst;

    streamed(T &dst) : m_dst(dst) {}

    void operator=(T const &val) const {
        details::stream_write(&m_dst, &val);
    }

    operator float() const {
        return m_dst;
    }
};

template <class T>
streamed(T &) -> streamed<T>;

template <class T, bool kEnabled = true>
struct streamed_array {
    T *m_data;

    streamed_array(T *data, int = 0) : m_data(data) {}

    std::conditional_t<kEnabled, streamed<T>, T &> operator[](std::size_t i) const {
        if constexpr (kEnabled) {
            return streamed<T>{m_data[i]};
        } else {
            return m_data[i];
        }
    }
};

template <class T>
streamed_array(T *) -> streamed_array<T>;

template <class T, class Enabled>
streamed_array(T *, Enabled) -> streamed_array<T, Enabled::value>;

}
}
