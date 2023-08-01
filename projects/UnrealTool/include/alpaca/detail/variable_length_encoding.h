#pragma once
#include <alpaca/detail/output_container.h>
#include <cstdint>
#include <utility>
#include <vector>

namespace alpaca {

namespace detail {

template <typename T> bool CHECK_BIT(T &value, uint8_t pos) {
  return ((value) & (T{1} << (pos)));
}

template <typename T> void SET_BIT(T &value, uint8_t pos) {
  value = value | T{1} << pos;
}

template <typename T> void RESET_BIT(T &value, uint8_t pos) {
  value = value & ~(T{1} << pos);
}

template <typename int_t, typename Container>
bool encode_varint_firstbyte_6(int_t &value, Container &output,
                               std::size_t &byte_index) {
  uint8_t octet = 0;
  if (value < 0) {
    value *= -1;
    SET_BIT(octet, 7);
  }
  // While more than 7 bits of data are left, occupy the last output byte
  // and set the next byte flag
  if (value > 63) {
    // Set the next byte flag
    octet |= ((uint8_t)(value & 63)) | 64;
    append(octet, output, byte_index);
    return true; // multibyte
  } else {
    octet |= ((uint8_t)(value & 63));
    append(octet, output, byte_index);
    return false; // no more bytes needed
  }
}

template <typename int_t, typename Container>
void encode_varint_6(int_t value, Container &output, std::size_t &byte_index) {
  // While more than 7 bits of data are left, occupy the last output byte
  // and set the next byte flag
  while (value > 63) {
    // Set the next byte flag
    append(((uint8_t)(value & 63)) | 64, output, byte_index);
    // Remove the seven bits we just wrote
    value >>= 6;
  }
  append(((uint8_t)value) & 63, output, byte_index);
}

template <typename int_t, typename Container>
typename std::enable_if<!std::is_same_v<Container, std::ifstream>, int_t>::type
decode_varint_firstbyte_6(Container &input, std::size_t &current_index,
                          bool &negative, bool &multibyte) {
  int octet = 0;
  int_t current = input[current_index];
  if (CHECK_BIT(current, 7)) {
    // negative number
    RESET_BIT(current, 7);
    negative = true;
  }
  if (CHECK_BIT(current, 6)) {
    RESET_BIT(current, 6);
    multibyte = true;
  }

  octet |= input[current_index++] & 63;
  return static_cast<int_t>(octet);
}

template <typename int_t, typename Container>
typename std::enable_if<std::is_same_v<Container, std::ifstream>, int_t>::type
decode_varint_firstbyte_6(Container &input, std::size_t &current_index,
                          bool &negative, bool &multibyte) {
  int octet = 0;

  // read byte from file stream
  char current_byte;
  input.read(&current_byte, 1);
  uint8_t byte = static_cast<uint8_t>(current_byte);

  int_t current = byte;
  if (CHECK_BIT(current, 7)) {
    // negative number
    RESET_BIT(current, 7);
    negative = true;
  }
  if (CHECK_BIT(current, 6)) {
    RESET_BIT(current, 6);
    multibyte = true;
  }

  octet |= byte & 63;
  current_index += 1;
  return static_cast<int_t>(octet);
}

template <typename int_t, typename Container>
typename std::enable_if<!std::is_same_v<Container, std::ifstream>, int_t>::type
decode_varint_6(Container &input, std::size_t &current_index) {
  int_t ret = 0;
  for (std::size_t i = 0; i < sizeof(int_t); ++i) {
    ret |= (static_cast<int_t>(input[current_index + i] & 63)) << (6 * i);
    // If the next-byte flag is set
    if (!(input[current_index + i] & 64)) {
      current_index += i + 1;
      break;
    }
  }
  return ret;
}

// ifstream version
template <typename int_t, typename Container>
typename std::enable_if<std::is_same_v<Container, std::ifstream>, int_t>::type
decode_varint_6(Container &input, std::size_t &current_index) {
  int_t ret = 0;
  for (std::size_t i = 0; i < sizeof(int_t); ++i) {

    // read byte from file stream
    char current_byte;
    input.read(&current_byte, 1);
    uint8_t byte = static_cast<uint8_t>(current_byte);

    ret |= (static_cast<int_t>(byte & 63)) << (6 * i);
    // If the next-byte flag is set
    if (!(byte & 64)) {
      current_index += i + 1;
      break;
    }
  }
  return ret;
}

template <typename int_t, typename Container>
void encode_varint_7(int_t value, Container &output, std::size_t &byte_index) {
  if (value < 0) {
    value *= 1;
  }
  // While more than 7 bits of data are left, occupy the last output byte
  // and set the next byte flag
  while (value > 127) {
    //|128: Set the next byte flag
    append(((uint8_t)(value & 127)) | 128, output, byte_index);
    // Remove the seven bits we just wrote
    value >>= 7;
  }
  append(((uint8_t)value) & 127, output, byte_index);
}

template <typename int_t, typename Container>
typename std::enable_if<!std::is_same_v<Container, std::ifstream>, int_t>::type
decode_varint_7(Container &input, std::size_t &current_index) {
  int_t ret = 0;
  for (std::size_t i = 0; i < sizeof(int_t); ++i) {
    ret |= (static_cast<int_t>(input[current_index + i] & 127)) << (7 * i);
    // If the next-byte flag is set
    if (!(input[current_index + i] & 128)) {
      current_index += i + 1;
      break;
    }
  }
  return ret;
}

// ifstream version
template <typename int_t, typename Container>
int_t decode_varint_7(std::ifstream &input, std::size_t &current_index) {
  int_t ret = 0;
  for (std::size_t i = 0; i < sizeof(int_t); ++i) {

    // read byte from file stream
    char current_byte;
    input.read(&current_byte, 1);
    uint8_t byte = static_cast<uint8_t>(current_byte);

    ret |= (static_cast<int_t>(byte & 127)) << (7 * i);
    // If the next-byte flag is set
    if (!(byte & 128)) {
      current_index += i + 1;
      break;
    }
  }
  return ret;
}

// Unsigned integer variable-length encoding functions
template <typename int_t, typename Container>
typename std::enable_if<std::is_integral_v<int_t> && !std::is_signed_v<int_t>,
                        void>::type
encode_varint(int_t value, Container &output, std::size_t &byte_index) {
  encode_varint_7<int_t>(value, output, byte_index);
}

template <typename int_t, typename Container>
typename std::enable_if<std::is_integral_v<int_t> && !std::is_signed_v<int_t>,
                        int_t>::type
decode_varint(Container &input, std::size_t &current_index) {
  return decode_varint_7<int_t, Container>(input, current_index);
}

// Signed integer variable-length encoding functions
template <typename int_t, typename Container>
typename std::enable_if<std::is_integral_v<int_t> && std::is_signed_v<int_t>,
                        void>::type
encode_varint(int_t value, Container &output, std::size_t &byte_index) {
  // first octet
  if (encode_varint_firstbyte_6<int_t>(value, output, byte_index)) {
    // rest of the octets
    encode_varint_7<int_t>(value, output, byte_index);
  }
}

template <typename int_t, typename Container>
typename std::enable_if<std::is_integral_v<int_t> && std::is_signed_v<int_t>,
                        int_t>::type
decode_varint(Container &input, std::size_t &current_index) {
  // decode first byte
  bool is_negative = false, multibyte = false;
  auto ret = decode_varint_firstbyte_6<int_t, Container>(
      input, current_index, is_negative, multibyte);

  // decode rest of the bytes
  // if continuation bit is set
  if (multibyte) {
    ret |= decode_varint_7<int_t, Container>(input, current_index);
  }

  if (is_negative) {
    ret *= -1;
  }

  return ret;
}

} // namespace detail

} // namespace alpaca
