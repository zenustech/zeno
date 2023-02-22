#ifndef ZENO_NETWORKTYPES_H
#define ZENO_NETWORKTYPES_H

#include <type_traits>
#include <string>
#include <array>

enum class ZBTControlPacketType : uint16_t {
    Start = 0x0000,
    AuthRequire,
    End,
    Max = 0xFFFF,
};

/**
 * Header struct of ZenoBridge TCP Packet
 */
struct ZBTPacketHeader {
    explicit ZBTPacketHeader(uint16_t inIndex, uint16_t inLength, ZBTControlPacketType inType)
        : marker(0xDA2C),
          index(inIndex),
          length(inLength),
          type(inType)
    {}

    /** Packet header marker */
    uint16_t marker;
    /** Loop index of packet index */
    uint16_t index;
    /** Length of bytes */
    uint16_t length;
    /** Route type of this packet */
    ZBTControlPacketType type;
};

/** bytes to spilt up the packet stream */
constexpr std::array<uint8_t, 2> g_packetSplit { 0x03, 0x04 };

/**
 * A packet looks like
 *  +------------+--------------+-------------+
    |    Index   |    Length    | Packet Type |
    |    2Byte   |    2Byte     |    4Byte    |
    +------------+--------------+-------------+
    |                                         |
    |                                         |
    |          Data                           |
    |                                         |
    |          Size equal to length bytes     |
    |                                         |
    |                                         |
    |                                         |
    +-----------------------------------------+
    |         0x03             0x04           |
    |                                         |
    +-----------------------------------------+
 */

#endif //ZENO_NETWORKTYPES_H
