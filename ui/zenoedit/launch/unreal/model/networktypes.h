#ifndef ZENO_NETWORKTYPES_H
#define ZENO_NETWORKTYPES_H

#include <type_traits>
#include <string>
#include <array>
#include <vector>

enum class ZBTControlPacketType : uint16_t {
    Start = 0x0000,
    AuthRequire,
    SendHeightField,
    RegisterSession,
    SendAuthToken,
    End,
    Max = 0xFFFF,
};

/** bytes to indicate start of packet */
constexpr std::array<uint8_t, 2> g_packetStart { 0xDA, 0x2C };

/** bytes to spilt up the packet stream */
constexpr std::array<uint8_t, 4> g_packetSplit { 0x03, 0x04, 0xA5, 0xF6 };

enum class ZBFileType : uint32_t {
    Start = 0,
    HeightField,
    End,
    Max = 0xFFFFFFFF,
};

/**
 * Header struct of ZenoBridge TCP Packet
* A packet looks like
*  +------------+--------------+-------------+
   |Marker+Index|    Length    | Packet Type |
   |    4Byte   |    2Byte     |    2Byte    |
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
struct alignas(8) ZBTPacketHeader {
    explicit ZBTPacketHeader(uint16_t inIndex, uint16_t inLength, ZBTControlPacketType inType, uint16_t inMarker = *reinterpret_cast<const uint16_t*>(g_packetStart.data()))
        : marker(inMarker),
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

// UDP
struct alignas(8) ZBUFileMessageHeader {
    ZBFileType type;
    uint32_t size, file_id;
    uint16_t total_part, part_id;
};

struct alignas(8) ZBUFileMessage {
    ZBUFileMessageHeader header;
    std::vector<uint8_t> data;
};

// Packet Bodies
struct alignas(8) ZPKSendToken {
    std::string token;

    template <typename T>
    void pack(T& pack) {
        pack(token);
    }
};

#endif //ZENO_NETWORKTYPES_H
