#ifndef ZENO_PACKETHANDLER_H
#define ZENO_PACKETHANDLER_H

#include <optional>
#include <map>
#include <vector>
#include "networktypes.h"

typedef std::optional<std::vector<uint8_t>> OutPacketBufferType;
typedef void(*ZBTPacketHandler)(const void*, const uint16_t, bool&, ZBTControlPacketType&, OutPacketBufferType&, uint16_t&);

template <
    ZBTControlPacketType InPacketType
>
struct PacketHandlerAnnotation {
    static ZBTPacketHandler handler;
};

struct PacketHandlerMap {
    static PacketHandlerMap& get() {
        static PacketHandlerMap sMap;
        return sMap;
    }

    void addHandler(const ZBTControlPacketType value, ZBTPacketHandler handler) {
        handlerMap.insert(std::make_pair(value, handler));
    }

    void tryCall(
        const ZBTControlPacketType inPacketType,
        void* inPacket,
        uint16_t inSize,
        bool& outHasRespond,
        ZBTControlPacketType& outPacketType,
        OutPacketBufferType& outRespondData,
        uint16_t& outDataSize
    ) {
        auto it = handlerMap.find(inPacketType);
        if (it != handlerMap.end()) {
            ZBTPacketHandler handler = it->second;
            handler(inPacket, inSize, outHasRespond, outPacketType, outRespondData, outDataSize);
        }
    }

private:
    std::map<ZBTControlPacketType, ZBTPacketHandler> handlerMap;
};


#define REG_PACKET_HANDLER(Name, PacketType, Block) \
    template<>                                                                                                  \
    ZBTPacketHandler PacketHandlerAnnotation<PacketType>::handler = [] (const void* inData, const uint16_t inSize, bool& bHasRespond, ZBTControlPacketType& outPacketType, OutPacketBufferType& outBuffer, uint16_t& outSize) Block;\
    static struct THIS_IS_NOT_START_WITH_StaticInitFor##Name {     \
        THIS_IS_NOT_START_WITH_StaticInitFor##Name() {             \
            PacketHandlerMap::get().addHandler(PacketType, PacketHandlerAnnotation<PacketType>::handler); \
        }                                    \
    } THIS_IS_NOT_START_WITH__StaticInitFor##Name;

#endif //ZENO_PACKETHANDLER_H
