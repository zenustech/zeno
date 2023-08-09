#pragma once
#include <alpaca/detail/crc32.h>
#include <alpaca/detail/options.h>

namespace alpaca {

namespace detail {

constexpr bool is_system_little_endian() {
  int32_t x = 0xaabbccdd;
  uint8_t y = static_cast<uint8_t>(x);
  return (y == 0xdd);
}

constexpr bool is_system_big_endian() {
  int32_t x = 0xaabbccdd;
  uint8_t y = static_cast<uint8_t>(x);
  return (y == 0xaa);
}

// Endian switching code taken from
// https://github.com/dfelinto/blender/blob/master/source/blender/blenlib/BLI_endian_switch_inline.h

/* NOTE: using a temp char to switch endian is a lot slower,
 * use bit shifting instead. */

/* *** 16 *** */
static inline void BLI_endian_switch_uint16(unsigned short *val) {
#if (defined(__GNUC__) &&                                                      \
     ((__GNUC__ * 100 + __GNUC_MINOR__) >= 408)) /* gcc4.8+ only */
  *val = __builtin_bswap16(*val);
#else
  unsigned short tval = *val;
  *val = (tval >> 8) | (tval << 8);
#endif
}

static inline void BLI_endian_switch_int16(short *val) {
  BLI_endian_switch_uint16((unsigned short *)val);
}

/* *** 32 *** */
static inline void BLI_endian_switch_uint32(unsigned int *val) {
#ifdef __GNUC__
  *val = __builtin_bswap32(*val);
#else
  unsigned int tval = *val;
  *val = ((tval >> 24)) | ((tval << 8) & 0x00ff0000) |
         ((tval >> 8) & 0x0000ff00) | ((tval << 24));
#endif
}

static inline void BLI_endian_switch_int32(int *val) {
  BLI_endian_switch_uint32((unsigned int *)val);
}

static inline void BLI_endian_switch_float(float *val) {
  BLI_endian_switch_uint32((unsigned int *)val);
}

/* *** 64 *** */
static inline void BLI_endian_switch_uint64(uint64_t *val) {
#ifdef __GNUC__
  *val = __builtin_bswap64(*val);
#else
  uint64_t tval = *val;
  *val = ((tval >> 56)) | ((tval << 40) & 0x00ff000000000000ll) |
         ((tval << 24) & 0x0000ff0000000000ll) |
         ((tval << 8) & 0x000000ff00000000ll) |
         ((tval >> 8) & 0x00000000ff000000ll) |
         ((tval >> 24) & 0x0000000000ff0000ll) |
         ((tval >> 40) & 0x000000000000ff00ll) | ((tval << 56));
#endif
}

static inline void BLI_endian_switch_int64(int64_t *val) {
  BLI_endian_switch_uint64((uint64_t *)val);
}

static inline void BLI_endian_switch_double(double *val) {
  BLI_endian_switch_uint64((uint64_t *)val);
}

enum class byte_order { little_endian, big_endian };

template <typename T, byte_order O> constexpr auto byte_swap(const T &value) {
  T result = value;

  if constexpr (O == byte_order::little_endian && is_system_little_endian()) {
    // do nothing
  } else if constexpr (O == byte_order::big_endian && is_system_big_endian()) {
    // do nothing
  } else if constexpr ((O == byte_order::little_endian &&
                        is_system_big_endian()) ||
                       (O == byte_order::big_endian &&
                        is_system_little_endian())) {
    // byte swap to match requested order
    if constexpr (std::is_same_v<T, uint16_t>) {
      BLI_endian_switch_uint16(&result);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      BLI_endian_switch_uint32(&result);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      BLI_endian_switch_uint64(&result);
    } else if constexpr (std::is_same_v<T, int16_t>) {
      BLI_endian_switch_int16(&result);
    } else if constexpr (std::is_same_v<T, int32_t>) {
      BLI_endian_switch_int32(&result);
    } else if constexpr (std::is_same_v<T, int64_t>) {
      BLI_endian_switch_int64(&result);
    } else if constexpr (std::is_same_v<T, float>) {
      BLI_endian_switch_float(&result);
    } else if constexpr (std::is_same_v<T, double>) {
      BLI_endian_switch_double(&result);
    }
  }

  return result;
}

template <options O, typename T>
void update_value_based_on_alpaca_endian_rules(T &value) {
  // if system is big endian
  // but big endian was not requested in options
  // default to little endian
  if constexpr (is_system_big_endian() && detail::little_endian<O>()) {
    value = byte_swap<T, byte_order::big_endian>(value);
  }
  // if system is little endian
  // but big_endian is requested
  else if constexpr (is_system_little_endian() && detail::big_endian<O>()) {
    value = byte_swap<T, byte_order::big_endian>(value);
  }
}

} // namespace detail

} // namespace alpaca