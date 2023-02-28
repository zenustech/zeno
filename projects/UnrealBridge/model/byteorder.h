//
// From https://gist.github.com/apathyboy/801594/801d14406a04b042f2ca33b9498e188b700a49c6
//

#ifndef ZENO_BYTEORDER_H
#define ZENO_BYTEORDER_H
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <limits>

namespace zeno {

namespace byteorder {
/// @TODO Remove quotes from around constexpr when visual studio begins
/// supporting that c++0x feature.
/*constexpr*/ bool inline is_big_endian() {
    uint16_t x = 1;
    return !(*reinterpret_cast<char*>(&x));
}

struct bit8_tag{};
struct bit16_tag{};
struct bit32_tag{};
struct bit64_tag{};

template<typename T> struct integer_traits;

template<>
struct integer_traits<int8_t> {
    typedef bit8_tag category;
};

template<>
struct integer_traits<uint8_t> {
    typedef bit8_tag category;
};

template<>
struct integer_traits<int16_t> {
    typedef bit16_tag category;
};

template<>
struct integer_traits<uint16_t> {
    typedef bit16_tag category;
};

template<>
struct integer_traits<int32_t> {
    typedef bit32_tag category;
};

template<>
struct integer_traits<uint32_t> {
    typedef bit32_tag category;
};

template<>
struct integer_traits<int64_t> {
    typedef bit64_tag category;
};

template<>
struct integer_traits<uint64_t> {
    typedef bit64_tag category;
};

template<typename T>
T swap_endian_(T value, bit8_tag) {
    return value;
}

template<typename T>
T swap_endian_(T value, bit16_tag) {
    return (value >> 8) | (value << 8);
}

template<typename T>
T swap_endian_(T value, bit32_tag) {
    return (value >> 24) |
           ((value & 0x00FF0000) >> 8) | ((value & 0x0000FF00) << 8) |
           (value << 24);
}

template<typename T>
T swap_endian_(T value, bit64_tag) {
    return (value  >> 56) |
#ifdef _WIN32
           ((value & 0x00FF000000000000) >> 40) |
           ((value & 0x0000FF0000000000) >> 24) |
           ((value & 0x000000FF00000000) >> 8)  |
           ((value & 0x00000000FF000000) << 8)  |
           ((value & 0x0000000000FF0000) << 24) |
           ((value & 0x000000000000FF00) << 40) |
#else
           ((value & 0x00FF000000000000LLU) >> 40) |
           ((value & 0x0000FF0000000000LLU) >> 24) |
           ((value & 0x000000FF00000000LLU) >> 8)  |
           ((value & 0x00000000FF000000LLU) << 8)  |
           ((value & 0x0000000000FF0000LLU) << 24) |
           ((value & 0x000000000000FF00LLU) << 40) |
#endif
           (value  << 56);
}
}

    /*! Swaps the endianness of the passed in value and returns the results.
    *
    * For standard integer types (any of the intX_t/uintX_t types)
    * specializations exist to ensure the fastest performance. All other types
    * are treated as char* and reversed.
    */
template<typename T>
T swap_endian(T value) {
    if (std::numeric_limits<T>::is_integer) {
        return byteorder::swap_endian_<T>(value, byteorder::integer_traits<T>::category());
    }

    unsigned char* tmp = reinterpret_cast<unsigned char*>(&value);
    std::reverse(tmp, tmp + sizeof(T));
    return value;
}

/*! Converts a value from host-byte order to little endian.
    *
    * Only works with integer types.
    *
    * \param value The value to convert to little endian
    * \return The value converted to endian order.
    */
template<typename T>
T host_to_little(T value) {
    static_assert(std::numeric_limits<T>::is_integer);
    return byteorder::is_big_endian() ? swap_endian(value) : value;
}

/*! Converts a value from host-byte order to big endian.
    *
    * Only works with integer types.
    *
    * \param value The value to convert to big endian
    * \return The value converted to endian order.
    */
template<typename T>
T host_to_big(T value) {
    static_assert(std::numeric_limits<T>::is_integer);
    return byteorder::is_big_endian() ? value : swap_endian(value);
}

/*! Converts a value from big endian to host-byte order.
    *
    * Only works with integer types.
    *
    * \param value The value to convert to host-byte order.
    * \return The value converted to host-byte order.
    */
template<typename T>
T big_to_host(T value) {
    static_assert(std::numeric_limits<T>::is_integer);
    return byteorder::is_big_endian() ? value : swap_endian(value);
}

/*! Converts a value from little endian to host-byte order.
    *
    * Only works with integer types.
    *
    * \param value The value to convert to host-byte order.
    * \return The value converted to host-byte order.
    */
template<typename T>
T little_to_host(T value) {
    static_assert(std::numeric_limits<T>::is_integer);
    return byteorder::is_big_endian() ? swap_endian(value) : value;
}

}
#endif //ZENO_BYTEORDER_H
